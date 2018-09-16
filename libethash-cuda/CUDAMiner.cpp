/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#undef min
#undef max
#include <ethash/ethash.hpp>

#include "CUDAMiner.h"
#include "ProgPoW_kernel.h"
#include <nvrtc.h>

using namespace std;
using namespace dev;
using namespace eth;

unsigned CUDAMiner::s_numInstances = 0;

vector<int> CUDAMiner::s_devices(MAX_MINERS, -1);

struct CUDAChannel : public LogChannel
{
    static const char* name() { return EthOrange "cu"; }
    static const int verbosity = 2;
};
#define cudalog clog(CUDAChannel)

CUDAMiner::CUDAMiner(FarmFace& _farm, unsigned _index)
  : Miner("cuda-", _farm, _index), m_light(getNumDevices())
{}

CUDAMiner::~CUDAMiner()
{
    stopWorking();
    kick_miner();
}

unsigned CUDAMiner::m_parallelHash = 4;
unsigned CUDAMiner::s_blockSize = CUDAMiner::c_defaultBlockSize;
unsigned CUDAMiner::s_gridSize = CUDAMiner::c_defaultGridSize;
unsigned CUDAMiner::s_numStreams = CUDAMiner::c_defaultNumStreams;
unsigned CUDAMiner::s_scheduleFlag = 0;

bool CUDAMiner::init(int epoch)
{
    try
    {
        if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
            while (s_dagLoadIndex < index)
                this_thread::sleep_for(chrono::milliseconds(100));
        unsigned device = s_devices[index] > -1 ? s_devices[index] : index;

        cnote << "Initialising miner " << index;

        auto numDevices = getNumDevices();
        if (numDevices == 0)
            return false;

        // use selected device
        m_device_num = device < numDevices - 1 ? device : numDevices - 1;
        m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
        m_hwmoninfo.indexSource = HwMonitorIndexSource::CUDA;
        m_hwmoninfo.deviceIndex = m_device_num;

        cudaDeviceProp device_props;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));

        cudalog << "Using device: " << device_props.name
                << " (Compute " + to_string(device_props.major) + "." +
                       to_string(device_props.minor) + ")";

        m_search_buf = new volatile search_results*[s_numStreams];
        m_streams = new cudaStream_t[s_numStreams];

        const auto& context = ethash::get_global_epoch_context(epoch);
        const auto lightNumItems = context.light_cache_num_items;
        const auto lightSize = ethash::get_light_cache_size(lightNumItems);
        const auto dagNumItems = context.full_dataset_num_items;
        const auto dagSize = ethash::get_full_dataset_size(dagNumItems);
        uint32_t dagWords   = (unsigned)(dagSize / ETHASH_MIX_BYTES);

        CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
        cudalog << "Set Device to current";
        if (dagNumItems != m_dag_size || !m_dag)
        {
            // Check whether the current device has sufficient memory every time we recreate the dag
            if (device_props.totalGlobalMem < dagSize)
            {
                cudalog << "CUDA device " << string(device_props.name)
                        << " has insufficient GPU memory. "
                        << FormattedMemSize(device_props.totalGlobalMem) << " of memory found, "
                        << FormattedMemSize(dagSize) << " of memory required";
                return false;
            }
            // We need to reset the device and recreate the dag
            cudalog << "Resetting device";
            CUDA_SAFE_CALL(cudaDeviceReset());
            CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
            CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
            // We need to reset the light and the Dag for the following code to reallocate
            // since cudaDeviceReset() frees all previous allocated memory
            m_light[m_device_num] = nullptr;
            m_dag = nullptr;
        }
        // create buffer for cache
        hash128_t* dag = m_dag;
        hash64_t* light = m_light[m_device_num];

        compileKernel(epoch, dagWords);

        if (!light)
        {
            cudalog << "Allocating light with size: " << FormattedMemSize(lightSize);
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), lightSize));
        }
        // copy lightData to device
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), context.light_cache, lightSize,
            cudaMemcpyHostToDevice));
        m_light[m_device_num] = light;

        if (dagNumItems != m_dag_size || !dag)  // create buffer for dag
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));

        set_constants(dag, dagNumItems, light, lightNumItems);

        if (dagNumItems != m_dag_size || !dag)
        {
            // create mining buffers
            cudalog << "Generating mining buffers";
            for (unsigned i = 0; i != s_numStreams; ++i)
            {
                CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(search_results)));
                CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
            }

            m_current_target = 0;

            if (!s_dagInHostMemory)
            {
                if ((m_device_num == s_dagCreateDevice) || (s_dagLoadMode != DAG_LOAD_MODE_SINGLE))
                {
                    cudalog << "Generating DAG for GPU #" << m_device_num
                            << " with dagSize: " << FormattedMemSize(dagSize) << " ("
                            << FormattedMemSize(device_props.totalGlobalMem - dagSize - lightSize)
                            << " left)";
                    auto startDAG = std::chrono::steady_clock::now();

                    ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0]);

                    cudalog << "Generated DAG for GPU" << m_device_num << " in: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startDAG)
                                   .count()
                            << " ms.";

                    if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
                    {
                        uint8_t* memoryDAG = new uint8_t[dagSize];
                        cudalog << "Copying DAG from GPU #" << m_device_num << " to host";
                        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagSize,
                            cudaMemcpyDeviceToHost));

                        s_dagInHostMemory = memoryDAG;
                    }
                }
                else
                {
                    while (!s_dagInHostMemory)
                        this_thread::sleep_for(chrono::milliseconds(100));
                    goto cpyDag;
                }
            }
            else
            {
            cpyDag:
                cudalog << "Copying DAG from host to GPU #" << m_device_num;
                const void* hdag = (const void*)s_dagInHostMemory;
                CUDA_SAFE_CALL(cudaMemcpy(
                    reinterpret_cast<void*>(dag), hdag, dagSize, cudaMemcpyHostToDevice));
            }
        }

        m_dag = dag;
        m_dag_size = dagNumItems;

        s_dagLoadIndex++;

        if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
        {
            if (s_dagLoadIndex >= s_numInstances && s_dagInHostMemory)
            {
                // all devices have loaded DAG, we can free now
                delete[] s_dagInHostMemory;
                s_dagInHostMemory = nullptr;
                cnote << "Freeing DAG from host";
            }
        }
    }
    catch (std::runtime_error const& _e)
    {
        cwarn << "Error CUDA mining: " << _e.what();
        if (s_exit)
            exit(1);
        return false;
    }
    return true;
}

void CUDAMiner::workLoop()
{
    WorkPackage current;
    current.header = h256{1u};

    try
    {
        while (!shouldStop())
        {
            if (is_mining_paused())
            {
                // cnote << "Mining is paused: Waiting for 3s.";
                std::this_thread::sleep_for(std::chrono::seconds(3));
                continue;
            }

            // take local copy of work since it may end up being overwritten.
            const WorkPackage w = work();

            // Take actions in proper order

            // No work ?
            if (!w || w.header == h256())
            {
                cnote << "No work. Pause for 3 s.";
                std::this_thread::sleep_for(std::chrono::seconds(3));
                continue;
            }
            // Epoch change ?
            else if (current.epoch != w.epoch)
            {
                if (!init(w.epoch))
                    break;
            }

            // Persist most recent job anyway. No need to do another
            // conditional check if they're different
            current = w;

            uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);
            uint64_t startN = current.startNonce;
            if (current.exSizeBits >= 0)
            {
                // this can support up to 2^c_log2Max_miners devices
                startN = current.startNonce |
                         ((uint64_t)index << (64 - LOG2_MAX_MINERS - current.exSizeBits));
            }
            else
            {
                startN = get_start_nonce();
            }

            // Eventually start searching
            search(current.header.data(), upper64OfBoundary, (current.exSizeBits >= 0), startN, w);
        }

        // Reset miner and stop working
        CUDA_SAFE_CALL(cudaDeviceReset());
    }
    catch (cuda_runtime_error const& _e)
    {
        cwarn << "GPU error: " << _e.what();
        if (s_exit)
        {
            cwarn << "Terminating.";
            exit(1);
        }
    }
}

void CUDAMiner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
}

void CUDAMiner::setNumInstances(unsigned _instances)
{
    s_numInstances = std::min<unsigned>(_instances, getNumDevices());
}

void CUDAMiner::setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount)
{
    for (unsigned i = 0; i < _selectedDeviceCount; i++)
        s_devices[i] = _devices[i];
}

unsigned CUDAMiner::getNumDevices()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver)
    {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            throw std::runtime_error{"No CUDA driver found"};
        throw std::runtime_error{"Insufficient CUDA driver: " + std::to_string(driverVersion)};
    }

    throw std::runtime_error{cudaGetErrorString(err)};
}

void CUDAMiner::listDevices()
{
    try
    {
        cout << "\nListing CUDA devices.\nFORMAT: [deviceID] deviceName\n";
        int numDevices = getNumDevices();
        for (int i = 0; i < numDevices; ++i)
        {
            cudaDeviceProp props;
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

            cout << "[" + to_string(i) + "] " + string(props.name) + "\n";
            cout << "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) +
                        "\n";
            cout << "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
            cout << "\tPci: " << setw(4) << setfill('0') << hex << props.pciDomainID << ':'
                 << setw(2) << props.pciBusID << ':' << setw(2) << props.pciDeviceID << '\n';
        }
    }
    catch (std::runtime_error const& err)
    {
        cwarn << "CUDA error: " << err.what();
        if (s_exit)
            exit(1);
    }
}

/// XXX
unsigned const CUDAMiner::c_defaultBlockSize = 512;
unsigned const CUDAMiner::c_defaultGridSize = 1024;  // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const CUDAMiner::c_defaultNumStreams = 2;

bool CUDAMiner::configureGPU(unsigned _blockSize, unsigned _gridSize, unsigned _numStreams,
    unsigned _scheduleFlag, unsigned _dagLoadMode, unsigned _dagCreateDevice, bool _noeval,
    bool _exit)
{
    s_dagLoadMode = _dagLoadMode;
    s_dagCreateDevice = _dagCreateDevice;
    s_exit = _exit;
    s_blockSize = _blockSize;
    s_gridSize = _gridSize;
    s_numStreams = _numStreams;
    s_scheduleFlag = _scheduleFlag;
    s_noeval = _noeval;
    // ProgPoW CPU validation is not implemented, override
    s_noeval = true;

    try
    {

        cudalog << "Using grid size: " << s_gridSize << ", block size: " << s_blockSize;

        // by default let's only consider the DAG of the first epoch
        const auto dagSize =
            ethash::get_full_dataset_size(ethash::calculate_full_dataset_num_items(0));
        int devicesCount = static_cast<int>(getNumDevices());
        for (int i = 0; i < devicesCount; i++)
        {
            if (s_devices[i] != -1)
            {
                int deviceId = min(devicesCount - 1, s_devices[i]);
                cudaDeviceProp props;
                CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
                if (props.totalGlobalMem >= dagSize)
                {
                    cudalog << "Found suitable CUDA device [" << string(props.name) << "] with "
                            << props.totalGlobalMem << " bytes of GPU memory";
                }
                else
                {
                    cudalog << "CUDA device " << string(props.name)
                            << " has insufficient GPU memory. " << props.totalGlobalMem
                            << " bytes of memory found < " << dagSize
                            << " bytes of memory required";
                    return false;
                }
            }
        }
        return true;
    }
    catch (runtime_error)
    {
        if (s_exit)
            exit(1);
        return false;
    }

    return true;
}

void CUDAMiner::setParallelHash(unsigned _parallelHash)
{
    m_parallelHash = _parallelHash;
}

#include <iostream>
#include <fstream>

void CUDAMiner::compileKernel(int epoch, uint64_t dag_words)
{
	const char* name = "progpow_search";

	std::string text = ProgPow::getKern((uint64_t)epoch, ProgPow::KERNEL_CUDA);
	text += std::string(ProgPoW_kernel, sizeof(ProgPoW_kernel));

	ofstream write;
	write.open("kernel.cu");
	write << text;
	write.close();

	nvrtcProgram prog;
	NVRTC_SAFE_CALL(
		nvrtcCreateProgram(
			&prog,         // prog
			text.c_str(),  // buffer
			"kernel.cu",    // name
			0,             // numHeaders
			NULL,          // headers
			NULL));        // includeNames

	NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, name));
	cudaDeviceProp device_props;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));
	std::string op_arch = "--gpu-architecture=compute_" + to_string(device_props.major) + to_string(device_props.minor);
	std::string op_dag = "-DPROGPOW_DAG_WORDS=" + to_string(dag_words);

	const char *opts[] = {
		op_arch.c_str(),
		op_dag.c_str(),
		"-lineinfo"
	};
	nvrtcResult compileResult = nvrtcCompileProgram(
		prog,  // prog
		3,     // numOptions
		opts); // options
	// Obtain compilation log from the program.
	size_t logSize;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	char *log = new char[logSize];
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    if (g_logVerbosity >= 6)
    {
        cudalog << "Compile log: " << log;
    }
	delete[] log;
	NVRTC_SAFE_CALL(compileResult);
	// Obtain PTX from the program.
	size_t ptxSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	char *ptx = new char[ptxSize];
	NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
	write.open("kernel.ptx");
	write << ptx;
	write.close();
	// Load the generated PTX and get a handle to the kernel.
	char *jitInfo = new char[32 * 1024];
	char *jitErr = new char[32 * 1024];
	CUjit_option jitOpt[] = {
		CU_JIT_INFO_LOG_BUFFER,
		CU_JIT_ERROR_LOG_BUFFER,
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_LOG_VERBOSE,
		CU_JIT_GENERATE_LINE_INFO
	};
	void *jitOptVal[] = {
		jitInfo,
		jitErr,
		(void*)(32 * 1024),
		(void*)(32 * 1024),
		(void*)(1),
		(void*)(1)
	};
	CU_SAFE_CALL(cuModuleLoadDataEx(&m_module, ptx, 6, jitOpt, jitOptVal));
    if (g_logVerbosity >= 6)
    {
        cudalog << "JIT info: \n" << jitInfo;
    }
    if (jitErr[0] != 0){
        cudalog << "JIT err: \n" << jitErr;
    }

	delete[] ptx;
	delete[] jitInfo;
	delete[] jitErr;
	// Find the mangled name
	const char* mangledName;
	NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, name, &mangledName));
	CU_SAFE_CALL(cuModuleGetFunction(&m_kernel, m_module, mangledName));
	cudalog << "Compiled " << mangledName;;
	// Destroy the program.
	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

void CUDAMiner::search(uint8_t const* header, uint64_t target, bool _ethStratum, uint64_t _startN, const dev::eth::WorkPackage& w)
{
    const uint16_t kReportingInterval = 512;  // Must be a power of 2 passes

    bool initialize = false;
    if (memcmp(&m_current_header, header, sizeof(hash32_t)))
    {
        m_current_header = *reinterpret_cast<hash32_t const *>(header);
        initialize = true;
    }
    if (m_current_target != target)
    {
        m_current_target = target;
        initialize = true;
    }
    if (_ethStratum)
    {
        if (initialize)
        {
            m_starting_nonce = 0;
            m_current_index = 0;
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            for (unsigned int i = 0; i < s_numStreams; i++)
                m_search_buf[i]->count = 0;
        }
        if (m_starting_nonce != _startN)
        {
            // reset nonce counter
            m_starting_nonce = _startN;
            m_current_nonce = m_starting_nonce;
        }
    }
    else
    {
        if (initialize)
        {
            m_current_nonce = get_start_nonce();
            m_current_index = 0;
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            for (unsigned int i = 0; i < s_numStreams; i++)
                m_search_buf[i]->count = 0;
        }
    }
    const uint32_t batch_size = s_gridSize * s_blockSize;
    while (true)
    {
        m_current_index++;
        m_current_nonce += batch_size;
        auto stream_index = m_current_index % s_numStreams;
        cudaStream_t stream = m_streams[stream_index];
        volatile search_results* buffer = m_search_buf[stream_index];
        uint32_t found_count = 0;
        uint64_t nonces[SEARCH_RESULTS];
        h256 mixes[SEARCH_RESULTS];
        uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;
        if (m_current_index >= s_numStreams)
        {
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            found_count = buffer->count;
            if (found_count) {
                buffer->count = 0;
                if (found_count > SEARCH_RESULTS)
                    found_count = SEARCH_RESULTS;
                for (unsigned int j = 0; j < found_count; j++) {
                    nonces[j] = nonce_base + buffer->result[j].gid;
					if (s_noeval)
                        memcpy(mixes[j].data(), (void *)&buffer->result[j].mix, sizeof(buffer->result[j].mix));
                }
            }
        }
        void *args[] = {&m_current_nonce, &m_current_header, &m_current_target, &m_dag, &buffer};
        CU_SAFE_CALL(cuLaunchKernel(m_kernel,
                                    s_gridSize, 1, 1,   // grid dim
                                    s_blockSize, 1, 1,  // block dim
                                    0,					// shared mem
                                    stream,				// stream
                                    args, 0));          // arguments
        if (m_current_index >= s_numStreams)
        {
            if (found_count)
            {
                for (uint32_t i = 0; i < found_count; i++)
                    if (s_noeval)
                        farm.submitProof(Solution{nonces[i], mixes[i], w, m_new_work});
                    else
                    {
                        Result r = EthashAux::eval(w.epoch, w.header, nonces[i]);
                        if (r.value < w.boundary)
                            farm.submitProof(Solution{nonces[i], r.mixHash, w, m_new_work});
                        else
                        {
                            farm.failedSolution();
                            cwarn
                                    << "GPU gave incorrect result! Lower OC if this happens frequently";
                        }
                    }
            }

            // stretch cuda passes to miniize the effects of
            // OS latency variability
            m_searchPasses++;
            if ((m_searchPasses & (kReportingInterval - 1)) == 0)
                updateHashRate(batch_size * kReportingInterval);
            bool t = true;
            if (m_new_work.compare_exchange_strong(t, false)) {
                if (g_logVerbosity >= 6)
                {
                    cudalog << "Switch time: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - workSwitchStart)
                                    .count()
                            << " ms.";
                }
                break;
            }
            if (shouldStop())
            {
                m_new_work.store(false, std::memory_order_relaxed);
                break;
            }
        }
    }
}
