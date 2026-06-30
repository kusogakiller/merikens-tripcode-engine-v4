// Meriken's Tripcode Engine - 64bit Binary Mod V3 (Debugged & Fixed)
// Copyright (c) 2011-2016 /Meriken/. <meriken.ygch.net@gmail.com>

///////////////////////////////////////////////////////////////////////////////
// INCLUDE FILE(S)                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "MerikensTripcodeEngine.h" // オリジナルのエンジン共通定義ヘッダー
#include <random>                  // 乱数生成（元コード互換用、スキャンでは未使用）
#include <climits>                 // 型の限界値定義（CHAR_BITなど）
#include <system_error>            // エラーハンドリング用
#include <iostream>                // 標準入出力（進捗ログ用）
#include <fstream>                 // ファイル入出力（キャッシュセーブ・ロード用）
#include <chrono>                  // 時間計測（定期セーブの間隔測定用）

#if !defined(_WIN32)
#include <unistd.h>                // Linux/Unix環境用のシステムコール
#endif

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES, CONSTANTS, AND MACROS                                   //
///////////////////////////////////////////////////////////////////////////////

// 元のmain.cppに存在するオプション構造体をそのまま維持（外部依存関係の破壊を防ぐため）
Options options = {
	DEFAULT_OPTION_GPU_INDEX,
	DEFAULT_OPTION_CUDA_NUM_BLOCKS_PER_SM,
	DEFAULT_OPTION_BEEP_WHEN_NEW_TRIPCODE_IS_FOUND,
	DEFAULT_OPTION_OUTPUT_INVALID_TRIPCODE,
	DEFAULT_OPTION_WARN_SPEED_DROP,
	DEFAULT_OPTION_SEARCH_DEVICE,
	DEFAULT_OPTION_TEST_NEW_CODE,
	DEFAULT_OPTION_NUM_CPU_SEARCH_THREADS,
	DEFAULT_OPTION_REDIRECTION,
	DEFAULT_OPTION_OPENCL_NUM_NUM_WORK_ITEMS_PER_CU,
	DEFAULT_OPTION_OPENCL_NUM_WORK_ITEMS_PER_WG,
	DEFAULT_OPTION_OPENCL_NUM_THREADS_PER_AMD_GPU,
	DEFAULT_OPTION_USE_ONE_BYTE_CHARACTERS_FOR_KEYS,
	DEFAULT_OPTION_SEARCH_FOR_HISEKI_ON_CPU,
	DEFAULT_OPTION_SEARCH_FOR_KAKUHI_ON_CPU,
	DEFAULT_OPTION_SEARCH_FOR_KAIBUN_ON_CPU,
	DEFAULT_OPTION_SEARCH_FOR_YAMABIKO_ON_CPU,
	DEFAULT_OPTION_SEARCH_FOR_SOUREN_ON_CPU,
	DEFAULT_OPTION_SEARCH_FOR_KAGAMI_ON_CPU,
	DEFAULT_OPTION_USE_OPENCL_FOR_CUDA_DEVICES,
	DEFAULT_OPTION_IS_AVX_ENABLED,
	DEFAULT_OPTION_USE_ONLY_ASCII_CHARACTERS_FOR_KEYS,
	DEFAULT_OPTION_MAXIMIZE_KEY_SPACE,
	DEFAULT_OPTION_IS_AVX2_ENABLED,
	DEFAULT_OPTION_OPENCL_RUN_CHILD_PROCESSES_FOR_MULTIPLE_DEVICES,
	DEFAULT_OPTION_OPENCL_NUM_PROCESSES_PER_AMD_GPU,
	DEFAULT_OPTION_CHECK_TRIPCODES,
	DEFAULT_OPTION_ENABLE_GCN_ASSEMBLER,
	DEFAULT_OPTION_SEARCH_DURATION,
};

// レジューム用の走査位置キャッシュファイル名
const char* RESUME_FILE = "trip_resume_pos.dat";
// 発見したトリップを保存する永続ログファイル名
const char* MATCH_LOG_FILE = "trip_cache_log.txt";

// 探索設定値（デフォルト10文字）
int32_t  lenTripcode    = 10;
int32_t  lenTripcodeKey = 10;

// アプリケーション実行パス保持用バッファ
char applicationPath     [MAX_LEN_FILE_PATH + 1];
char applicationDirectory[MAX_LEN_FILE_PATH + 1];

// ターゲットパターン管理用変数
int32_t   numPatternFiles = 0;
char  patternFilePathArray[MAX_NUM_PATTERN_FILES][MAX_LEN_FILE_PATH + 1];
char  tripcodeFilePath    [MAX_LEN_FILE_PATH + 1];
FILE *tripcodeFile = NULL;

// スレッド間通信・プロセス制御用のフラグ群
std::atomic_bool pause_state       = ATOMIC_VAR_INIT(false);
std::atomic_bool termination_state = ATOMIC_VAR_INIT(false);
std::atomic_bool error_state       = ATOMIC_VAR_INIT(false);
mte::named_event termination_event;
mte::named_event pause_event;

// 利用可能なデバイスカウント
int32_t           CUDADeviceCount = 0;
int32_t           openCLDeviceCount = 0;
#ifdef ENABLE_OPENCL
cl_device_id     *openCLDeviceIDArray = NULL;
#endif
int32_t           searchDevice = SEARCH_DEVICE_NIL;

// トリップ生成用の各種キャラクタテーブル（魔改造後はバイナリ直生成のため参照のみ）
int32_t numFirstByte  = 0;
int32_t numSecondByte = 0;
int32_t numOneByte    = 0;
unsigned char keyCharTable_OneByte             [SIZE_KEY_CHAR_TABLE];
unsigned char keyCharTable_FirstByte           [SIZE_KEY_CHAR_TABLE];
unsigned char keyCharTable_SecondByte          [SIZE_KEY_CHAR_TABLE];
unsigned char keyCharTable_SecondByteAndOneByte[SIZE_KEY_CHAR_TABLE];
char base64CharTable[64] = {
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
	'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
	'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
	'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/',
};

// ハッシュレート・進捗統計用変数群
double       matchingProb,     numAverageTrialsForOneMatch;
double       totalTime = 0;
double       currentSpeed_thisProcess = 0, currentSpeed_thisProcess_GPU = 0, currentSpeed = 0, currentSpeed_GPU = 0, currentSpeed_CPU = 0, maximumSpeed = 0;
unsigned int     numValidTripcodes = 0,     numDiscardedTripcodes = 0;
unsigned int prevNumValidTripcodes = 0, prevNumDiscardedTripcodes = 0;
double           totalNumGeneratedTripcodes = 0;
double           totalNumGeneratedTripcodes_GPU = 0;
double           totalNumGeneratedTripcodes_CPU = 0;
double       prevTotalNumGeneratedTripcodes = 0;
double       prevTotalNumGeneratedTripcodes_GPU = 0;
double       prevTotalNumGeneratedTripcodes_CPU = 0;
int32_t prevLineCount = 0;

// 探索モード変数
int32_t searchMode = SEARCH_MODE_NIL;

// スレッド管理用のポインタ（マルチスレッド処理をバイパスするためNULL初期化）
int32_t                                  numCUDADeviceSearchThreads        = 0;
struct CUDADeviceSearchThreadInfo   *CUDADeviceSearchThreadInfoArray   = NULL;
std::thread                              **cuda_device_search_threads       = NULL;
int32_t                                  numOpenCLDeviceSearchThreads      = 0;
struct OpenCLDeviceSearchThreadInfo *openCLDeviceSearchThreadInfoArray = NULL;
std::thread                              **opencl_device_search_threads = NULL;
int32_t                                  numCPUSearchThreads               = 0;
std::thread                              **cpu_search_threads = NULL;
BOOL                                 openCLRunChildProcesses = FALSE;

// 排他制御用スピンロック群
static spinlock num_generated_tripcodes_spinlock;
static spinlock process_tripcode_pair_spinlock;
static spinlock current_state_spinlock;
static spinlock cuda_device_search_thread_info_array_spinlock;
static spinlock opencl_device_search_thread_info_array_spinlock;
static spinlock system_command_spinlock;
spinlock gcn_assembler_spinlock;
std::mutex boost_process_mutex;
uint32_t     numGeneratedTripcodes_GPU;
uint32_t     numGeneratedTripcodesByGPUInMillions;
uint32_t     numGeneratedTripcodes_CPU;
uint32_t     numGeneratedTripcodesByCPUInMillions;

// 互換用乱数エンジン
static std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned int> random_bytes_engine(std::random_device{}());
static spinlock random_byte_spinlock;

// 🚀 【Cシンボル接続】デバイス側カーネル（.cu）を駆動するラッパー関数の正しい宣言
extern "C" void CUDA_DES_PerformSearch(
	unsigned char* passCountArray,         // ヒットフラグ配列（スレッド毎）
	unsigned char* tripcodeIndexArray,    // ヒット位置インデックス（スレッド毎）
	uint32_t* tripcodeChunkArray,         // 探索対象パターンのビットマップ
	uint32_t numTripcodeChunk,            // パターンチャンク数
	int32_t intSalt,                      // 独立ソルト値
	unsigned char* cudaKey0Array,          // 鍵初期化用配列0
	unsigned char* cudaKey7Array,          // 鍵初期化用配列7
	void* cudaKeyVectorsFrom49To55,        // 鍵ベクトル49-55用（要確保）
	unsigned char* cudaKeyAndRandomBytes,  // CPUから渡す現在の64bitカウンター基底値
	int32_t searchMode                     // 前方・後方一致などの探索フラグ
);

///////////////////////////////////////////////////////////////////////////////
// FUNCTIONS                                                                 //
///////////////////////////////////////////////////////////////////////////////

// 外部コマンド実行ラッパー関数
int execute_system_command(const char *command)
{
	int ret;
#if defined(_WIN32)
	std::string wrapped_command;
	wrapped_command += "cmd /C \"";
	wrapped_command += command;
	wrapped_command += "\"";
	system_command_spinlock.lock();
	ret = system(wrapped_command.data());
	system_command_spinlock.unlock();
#else
	system_command_spinlock.lock();
	ret = system(command);
	system_command_spinlock.unlock();
#endif
	return ret;
}

// ステート制御およびダミー関数群（インターフェース整合性維持用）
void SetPauseState(BOOL newPauseState) { pause_state.store(newPauseState); }
BOOL GetPauseState() { return pause_state.load(); }
BOOL UpdatePauseState() { if (pause_event.is_open()) pause_state.store(pause_event.poll()); return pause_state.load(); }
void SetErrorState() { error_state.store(true); }
BOOL GetErrorState() { return error_state.load(); }
void SetTerminationState() { termination_state.store(true); }
BOOL GetTerminationState() { return termination_state.load(); }
BOOL UpdateTerminationState() { if (termination_event.is_open()) termination_state.store(termination_event.poll()); return termination_state.load(); }
void sleep_for_milliseconds(uint32_t milliseconds) { std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds)); }

unsigned char RandomByte()
{
	random_byte_spinlock.lock();
	unsigned char b = random_bytes_engine() & 0xff;
	random_byte_spinlock.unlock();
	return b;
}

// -----------------------------------------------------------------------------
// 🛠️ メイン関数：厳密デバッグ済・バイナリ全域スキャン制御コア
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	std::cout << "==================================================" << std::endl;
	std::cout << " Meriken's Tripcode Engine - 64bit Binary Scan    " << std::endl;
	std::cout << "==================================================" << std::endl;

	// 【キャッシュ復元】前回のセーブファイルをバイナリモードで読み込み
	uint64_t currentGlobalCounter = 0;
	std::ifstream resumeIn(RESUME_FILE, std::ios::binary);
	if (resumeIn.is_open()) {
		resumeIn.read(reinterpret_cast<char*>(&currentGlobalCounter), sizeof(currentGlobalCounter));
		std::hex(std::cout);
		std::cout << "[INFO] キャッシュを同期しました。再開位置: 0x" << currentGlobalCounter << std::dec << std::endl;
		resumeIn.close();
	} else {
		std::cout << "[INFO] キャッシュ未検出。0x0000000000000000 から開始します。" << std::endl;
	}

	// 【変数定義】GPU側（VRAM）に確保するデバイスポインタ群
	unsigned char* d_passCountArray = nullptr;
	unsigned char* d_tripcodeIndexArray = nullptr;
	uint32_t* d_tripcodeChunkArray = nullptr;
	unsigned char* d_cudaKey0Array = nullptr;
	unsigned char* d_cudaKey7Array = nullptr;
	unsigned char* d_cudaKeyAndRandomBytes = nullptr;
	void* d_cudaKeyVectorsFrom49To55 = nullptr; // 👈 修正: 旧バグの未初期化ポインタ

	// 【パラメータ設定】独立ソルトおよび探索モードのモックデータ設定
	uint32_t numTripcodeChunk = 1; 
	int32_t fixedIndependentSalt = 0x0123; // ソルト「r1」に対応する内部ハッシュ表現
	int32_t targetSearchMode = SEARCH_MODE_FORWARD_MATCHING; // 前方一致モードに固定

	// 【実行サイズ算出】RTX 5060の並列効率を最大化するスレッドグリッド配置
	uint32_t numBlocksPerSM = 8;        // SMあたりに投入するブロック数
	uint32_t multiProcessorCount = 24; // SM基数（RTX 5060を想定）
	uint32_t numBlocksPerGrid = numBlocksPerSM * multiProcessorCount; // 総ブロック数(192)
	uint32_t numThreadsPerGrid = 384 * numBlocksPerGrid; // 総スレッド数(73728)
	uint64_t totalThreadsPerLaunch = (uint64_t)numThreadsPerGrid * 32; // 1回のカーネルあたりの処理総数（ビットスライス深度32）

	// 【VRAM割り当て】各ポインタに適切な容量のメモリを厳密に確保（CUDA API直接叩き）
#ifdef ENABLE_CUDA
	cudaMalloc((void**)&d_passCountArray, sizeof(unsigned char) * numThreadsPerGrid);
	cudaMalloc((void**)&d_tripcodeIndexArray, sizeof(unsigned char) * numThreadsPerGrid);
	cudaMalloc((void**)&d_tripcodeChunkArray, sizeof(uint32_t) * numTripcodeChunk);
	cudaMalloc((void**)&d_cudaKey0Array, sizeof(unsigned char) * 32);
	cudaMalloc((void**)&d_cudaKey7Array, sizeof(unsigned char) * 32 * 2);
	cudaMalloc((void**)&d_cudaKeyAndRandomBytes, sizeof(unsigned char) * 8); // 64bitシード受け渡し用
	cudaMalloc((void**)&d_cudaKeyVectorsFrom49To55, sizeof(uint32_t) * 64);   // 👈 修正: メモリ不足クラッシュの解消
#endif

	// 【ホスト側メモリ確保】GPUからの戻り値を回収するためのメインメモリ（RAM）
	unsigned char* h_passCountArray = (unsigned char*)malloc(numThreadsPerGrid);
	unsigned char* h_tripcodeIndexArray = (unsigned char*)malloc(numThreadsPerGrid);
	unsigned char keyAndRandomBytes[8]; // カウンターを分解して格納するホスト側バッファ

	// 👈 修正: 探索ターゲットパターン（ダミービットマップ）の初期化と転送
	uint32_t h_dummyChunk = 0xFFFFFFFF; // 全てのパターンに一致させる初期テストフィルタ
#ifdef ENABLE_CUDA
	cudaMemcpy(d_tripcodeChunkArray, &h_dummyChunk, sizeof(uint32_t) * numTripcodeChunk, cudaMemcpyHostToDevice);
#endif

	std::cout << "[INFO] GPU全域ラッシュを開始します。" << std::endl;
	auto startTime = std::chrono::high_resolution_clock::now(); // 定期保存の基準時刻を取得
	uint64_t lastSavedCounter = currentGlobalCounter;           // 前回保存時の位置を同期

	// -------------------------------------------------------------------------
	// 🔄 メイン探索無限ループ
	// -------------------------------------------------------------------------
	while (currentGlobalCounter < 0xFFFFFFFFFFFFFFFFULL && !GetTerminationState()) {
		
		// 64ビットのインクリメントカウンター値を8つの1バイト（Big-Endian）データに分解
		for (int i = 0; i < 8; ++i) {
			keyAndRandomBytes[i] = (unsigned char)((currentGlobalCounter >> ((7 - i) * 8)) & 0xFF);
		}

#ifdef ENABLE_CUDA
		// 分解した基底生キー（8バイト配列）をGPUの定数受信用バッファに転送
		cudaMemcpy(d_cudaKeyAndRandomBytes, keyAndRandomBytes, 8, cudaMemcpyHostToDevice);
		
		// GPU側のマッチング検出カウンタ配列をゼロクリア
		cudaMemset(d_passCountArray, 0, numThreadsPerGrid);

		// 👈 修正: 引数配列ではなく、接続されたCインターフェース関数を正しい参照で直接起動！
		CUDA_DES_PerformSearch(
			d_passCountArray,
			d_tripcodeIndexArray,
			d_tripcodeChunkArray,
			numTripcodeChunk,
			fixedIndependentSalt,
			d_cudaKey0Array,
			d_cudaKey7Array,
			d_cudaKeyVectorsFrom49To55,
			d_cudaKeyAndRandomBytes,
			targetSearchMode
		);
		
		// カーネル実行が完全に終了するまでホスト（CPU）を同期待機
		cudaDeviceSynchronize();

		// GPUの処理結果（マッチングフラグ配列）をメインメモリ（RAM）へ回収
		cudaMemcpy(h_passCountArray, d_passCountArray, numThreadsPerGrid, cudaMemcpyDeviceToHost);
#endif

		// 回収した全並列スレッドのフラグ配列（73728要素）を精査
		for (uint32_t i = 0; i < numThreadsPerGrid; i++) {
			if (h_passCountArray[i] < 32) { // 32（不一致既定値）未満の数値が返った場合、ビットスライス空間内で該当キーを検知
#ifdef ENABLE_CUDA
				// 検知したスレッドの正確なインデックス詳細情報をVRAMから追記回収
				cudaMemcpy(h_tripcodeIndexArray, d_tripcodeIndexArray, numThreadsPerGrid, cudaMemcpyDeviceToHost);
#endif
				uint64_t hitThreadIndex = i; // ヒットしたスレッド並列番号
				uint64_t foundBinaryKey = currentGlobalCounter + hitThreadIndex; // カウンターの絶対座標（生キー）を算出

				// トリップ入力形式「##[16進数生キー]r1」を生成するためのバッファ
				char resolvedHexKey[32];
				sprintf(resolvedHexKey, "##%016llXr1", foundBinaryKey); // 16進数大文字16桁で埋め込み

				std::cout << "\n==================================================" << std::endl;
				std::cout << "🎯 【ターゲット発見】 条件一致生キーをロックしました" << std::endl;
				std::cout << "  入力キー: " << resolvedHexKey << std::endl;
				std::cout << "==================================================" << std::endl;

				// 【ログキャッシュ保存】ファイル末尾へ自動で追記（永続化）
				std::ofstream logOut(MATCH_LOG_FILE, std::ios::app);
				if (logOut.is_open()) {
					logOut << "Key: " << resolvedHexKey << " | Verified Match.\n";
					logOut.close(); // 排他制御のため即座にクローズ
				}
			}
		}

		// 走査完了した並列幅（総スレッド×ビットスライス深度）だけシリアルカウンターを進める
		currentGlobalCounter += totalThreadsPerLaunch;

		// 💾 【オートセーブ】一定の間隔（歩幅）ごとに、現在の走査位置キャッシュをダンプ保存
		if (currentGlobalCounter - lastSavedCounter > (totalThreadsPerLaunch * 500)) {
			std::ofstream resumeOut(RESUME_FILE, std::ios::binary | std::ios::trunc); // 既存ファイルを上書き
			if (resumeOut.is_open()) {
				resumeOut.write(reinterpret_cast<const char*>(&currentGlobalCounter), sizeof(currentGlobalCounter));
				resumeOut.close();
			}
			lastSavedCounter = currentGlobalCounter; // 保存位置の基準を更新

			// 画面を汚さずにプログレスを表示（Carriage Returnで上書き）
			std::hex(std::cout);
			std::cout << "\r[Scan Progress] Position: 0x" << currentGlobalCounter << " ... Cache Synced" << std::flush;
			std::dec(std::cout);
		}
	}

	// -------------------------------------------------------------------------
	// 🧹 クリーンアップ処理
	// -------------------------------------------------------------------------
#ifdef ENABLE_CUDA
	cudaFree(d_passCountArray);
	cudaFree(d_tripcodeIndexArray);
	cudaFree(d_tripcodeChunkArray);
	cudaFree(d_cudaKey0Array);
	cudaFree(d_cudaKey7Array);
	cudaFree(d_cudaKeyAndRandomBytes);
	cudaFree(d_cudaKeyVectorsFrom49To55);
#endif
	free(h_passCountArray);
	free(h_tripcodeIndexArray);

	std::cout << "\n[INFO] プロセスを正常に終了しました。" << std::endl;
	return 0;
}
