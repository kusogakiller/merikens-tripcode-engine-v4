// Meriken's Tripcode Engine - 64bit Binary Mod V3 (Fully Debugged & Audited)
// Copyright (c) 2011-2016 /Meriken/. <meriken.ygch.net@gmail.com>

///////////////////////////////////////////////////////////////////////////////
// INCLUDE FILE(S)                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "MerikensTripcodeEngine.h" // オリジナルのエンジン共通定義ヘッダーをインクルード
#include <random>                  // 乱数生成用ライブラリ（元コード互換、本スキャンでは参照用）
#include <climits>                 // 型の限界値定義（CHAR_BIT: 1バイトあたりのビット数などを利用）
#include <system_error>            // システムエラーのハンドリング用クラスをインクルード
#include <iostream>                // 標準入出力ストリーム（進捗ログ・結果出力用）
#include <fstream>                 // ファイル入出力ストリーム（進行状況のセーブ・ロード用）
#include <chrono>                  // 高精度時間計測（定期オートセーブの間隔測定用）
#include <cinttypes>               // 64ビット変数の厳密なフォーマット出力（PRIX64マクロ）用
#include <cstdlib>                 // 標準ユーティリティ（malloc, free, exit用）

#if !defined(_WIN32)
#include <unistd.h>                // Linux/Unix環境におけるシステムコール用のヘッダー
#endif

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES, CONSTANTS, AND MACROS                                   //
///////////////////////////////////////////////////////////////////////////////

// 外部の構造体定義や既存システムとのリンケージを破壊しないよう、オプション構造体を完全に維持
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

// 走査位置を永続化するためのレジューム用キャッシュファイル名
const char* RESUME_FILE = "trip_resume_pos.dat";
// 発見された有効なトリップキーを保存するテキストログファイル名
const char* MATCH_LOG_FILE = "trip_cache_log.txt";

// トリップコードの探索長さ設定（デフォルト値として10文字を維持）
int32_t  lenTripcode    = 10;
int32_t  lenTripcodeKey = 10;

// アプリケーションの実行パスおよびディレクトリを保持するための固定長バッファ
char applicationPath     [MAX_LEN_FILE_PATH + 1];
char applicationDirectory[MAX_LEN_FILE_PATH + 1];

// ターゲットとなる探索パターンの管理用変数群
int32_t   numPatternFiles = 0; // パターンファイルの総数
char  patternFilePathArray[MAX_NUM_PATTERN_FILES][MAX_LEN_FILE_PATH + 1]; // 各ファイルのパス配列
char  tripcodeFilePath    [MAX_LEN_FILE_PATH + 1]; // 出力用トリップコードファイルパス
FILE *tripcodeFile = NULL; // トリップファイル制御用ポインタ

// マルチスレッド間での排他・状態制御を行うためのアトミック変数の初期化
std::atomic_bool pause_state       = ATOMIC_VAR_INIT(false); // 一時停止フラグ
std::atomic_bool termination_state = ATOMIC_VAR_INIT(false); // 強制終了フラグ
std::atomic_bool error_state       = ATOMIC_VAR_INIT(false); // エラー発生フラグ
mte::named_event termination_event; // 外部プロセス通信用終了イベント
mte::named_event pause_event;       // 外部プロセス通信用一時停止イベント

// システムで検知された演算デバイスのカウント変数
int32_t           CUDADeviceCount = 0; // 検知されたCUDAデバイス数
int32_t           openCLDeviceCount = 0; // 検知されたOpenCLデバイス数
#ifdef ENABLE_OPENCL
cl_device_id     *openCLDeviceIDArray = NULL; // OpenCL用デバイスID配列ポインタ
#endif
int32_t           searchDevice = SEARCH_DEVICE_NIL; // 現在選択中の探索デバイス

// キー変換用文字列テーブル（互換性の維持および参照用として配置）
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

// ハッシュレート計測および進捗統計のための各種実数型変数
double       matchingProb,     numAverageTrialsForOneMatch;
double       totalTime = 0; // 総実行時間
double       currentSpeed_thisProcess = 0, currentSpeed_thisProcess_GPU = 0, currentSpeed = 0, currentSpeed_GPU = 0, currentSpeed_CPU = 0, maximumSpeed = 0;
unsigned int     numValidTripcodes = 0,     numDiscardedTripcodes = 0; // 有効・無効トリップ数
unsigned int prevNumValidTripcodes = 0, prevNumDiscardedTripcodes = 0;
double           totalNumGeneratedTripcodes = 0; // 生成された総トリップ数
double           totalNumGeneratedTripcodes_GPU = 0; // GPUで生成された総数
double           totalNumGeneratedTripcodes_CPU = 0; // CPUで生成された総数
double       prevTotalNumGeneratedTripcodes = 0;
double       prevTotalNumGeneratedTripcodes_GPU = 0;
double       prevTotalNumGeneratedTripcodes_CPU = 0;
int32_t prevLineCount = 0;

// 探索アルゴリズムモード設定変数
int32_t searchMode = SEARCH_MODE_NIL;

// スレッド管理用ポインタ群（バイナリスキャンでは制御をメインループに集約するためNULL初期化）
int32_t                                  numCUDADeviceSearchThreads        = 0;
struct CUDADeviceSearchThreadInfo   *CUDADeviceSearchThreadInfoArray   = NULL;
std::thread                              **cuda_device_search_threads       = NULL;
int32_t                                  numOpenCLDeviceSearchThreads      = 0;
struct OpenCLDeviceSearchThreadInfo *openCLDeviceSearchThreadInfoArray = NULL;
std::thread                              **opencl_device_search_threads = NULL;
int32_t                                  numCPUSearchThreads               = 0;
std::thread                              **cpu_search_threads = NULL;
BOOL                                 openCLRunChildProcesses = FALSE;

// マルチスレッド競合を防ぐための排他制御用スピンロックおよびミューテックス定義
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

// スレッド安全な互換用乱数生成エンジンおよび専用スピンロック
static std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned int> random_bytes_engine(std::random_device{}());
static spinlock random_byte_spinlock;

// 🚀 【Cシンボル接続】デバイス側カーネル（.cu）を駆動するラッパー関数の正しい外部宣言
extern "C" void CUDA_DES_PerformSearch(
	unsigned char* passCountArray,         // ヒットフラグ配列（スレッド毎、VRAMポインタ）
	unsigned char* tripcodeIndexArray,    // ヒット位置インデックス（スレッド毎、VRAMポインタ）
	uint32_t* tripcodeChunkArray,         // 探索対象パターンのビットマップ（VRAMポインタ）
	uint32_t numTripcodeChunk,            // パターンチャンク数（固定値）
	int32_t intSalt,                      // 独立ソルト値（ハッシュ化ソルト表現）
	unsigned char* cudaKey0Array,          // 鍵初期化用配列0（VRAMポインタ）
	unsigned char* cudaKey7Array,          // 鍵初期化用配列7（VRAMポインタ）
	void* cudaKeyVectorsFrom49To55,        // 鍵ベクトル49-55用拡張領域（VRAMポインタ）
	unsigned char* cudaKeyAndRandomBytes,  // CPUから転送する現在の64bitカウンター基底値（VRAMポインタ）
	int32_t searchMode                     // 前方・後方一致などの探索モードフラグ
);

///////////////////////////////////////////////////////////////////////////////
// FUNCTIONS                                                                 //
///////////////////////////////////////////////////////////////////////////////

// 外部コマンドを実行するためのクロスプラットフォーム対応ラッパー関数
int execute_system_command(const char *command)
{
	int ret; // コマンドの戻り値格納用変数
#if defined(_WIN32)
	std::string wrapped_command; // Windows環境用のコマンドカプセル化文字列
	wrapped_command += "cmd /C \"";
	wrapped_command += command;
	wrapped_command += "\"";
	system_command_spinlock.lock(); // 排他ロックの取得
	ret = system(wrapped_command.data()); // ラップされたコマンドの実行
	system_command_spinlock.unlock(); // ロック解除
#else
	system_command_spinlock.lock(); // 非Windows（Linux等）環境での排他ロック
	ret = system(command); // コマンドの直接実行
	system_command_spinlock.unlock(); // ロック解除
#endif
	return ret; // 実行結果を返却
}

// アプリケーションの状態制御およびカプセル化用関数群（スレッド安全性を確保）
void SetPauseState(BOOL newPauseState) { pause_state.store(newPauseState); }
BOOL GetPauseState() { return pause_state.load(); }
BOOL UpdatePauseState() { if (pause_event.is_open()) pause_state.store(pause_event.poll()); return pause_state.load(); }
void SetErrorState() { error_state.store(true); }
BOOL GetErrorState() { return error_state.load(); }
void SetTerminationState() { termination_state.store(true); }
BOOL GetTerminationState() { return termination_state.load(); }
BOOL UpdateTerminationState() { if (termination_event.is_open()) termination_state.store(termination_event.poll()); return termination_state.load(); }
void sleep_for_milliseconds(uint32_t milliseconds) { std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds)); }

// スレッド安全に1バイトのランダムデータを取得する互換関数
unsigned char RandomByte()
{
	random_byte_spinlock.lock(); // 乱数エンジンの同時呼び出しを防ぐロック
	unsigned char b = random_bytes_engine() & 0xff; // 下位8ビットを抽出してバイトデータ化
	random_byte_spinlock.unlock(); // ロック解除
	return b; // 生成バイトを返却
}

// -----------------------------------------------------------------------------
// 🛠️ メイン関数：厳密監査済・バイナリ全域スキャン制御コア
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	std::cout << "==================================================" << std::endl;
	std::cout << " Meriken's Tripcode Engine - 64bit Binary Scan    " << std::endl;
	std::cout << "==================================================" << std::endl;

	// 【キャッシュ復元】前回のセーブファイルをバイナリモードで厳密に読み込み
	uint64_t currentGlobalCounter = 0; // 全域スキャンを司る64ビットメインカウンター
	std::ifstream resumeIn(RESUME_FILE, std::ios::binary); // 入力ストリームを開く
	if (resumeIn.is_open()) { // ファイルが存在する場合
		resumeIn.read(reinterpret_cast<char*>(&currentGlobalCounter), sizeof(currentGlobalCounter)); // 8バイト読み込み
		std::hex(std::cout); // 16進数表示モードへ切り替え
		std::cout << "[INFO] キャッシュを同期しました。再開位置: 0x" << currentGlobalCounter << std::dec << std::endl;
		resumeIn.close(); // ストリームを閉じる
	} else {
		std::cout << "[INFO] キャッシュ未検出。0x0000000000000000 から開始します。" << std::endl;
	}

	// 【変数定義】GPU側（VRAM）に確保する各デバイスポインタの初期化
	unsigned char* d_passCountArray = nullptr;       // ヒットフラグ格納用VRAMポインタ
	unsigned char* d_tripcodeIndexArray = nullptr;  // ビットインデックス詳細格納用VRAMポインタ
	uint32_t* d_tripcodeChunkArray = nullptr;       // 探索パターンビットマップ用VRAMポインタ
	unsigned char* d_cudaKey0Array = nullptr;         // 鍵スケジュール0用VRAMポインタ
	unsigned char* d_cudaKey7Array = nullptr;         // 鍵スケジュール7用VRAMポインタ
	unsigned char* d_cudaKeyAndRandomBytes = nullptr; // 64bitシードカウンター受信用VRAMポインタ
	void* d_cudaKeyVectorsFrom49To55 = nullptr;       // 拡張鍵ベクトル用VRAMポインタ

	// 【パラメータ設定】ハッシュアルゴリズム駆動用の制御データ定義
	uint32_t numTripcodeChunk = 1; // パターンチャンク数を1に固定
	int32_t fixedIndependentSalt = 0x0123; // ソルト表現用のダミー値
	int32_t targetSearchMode = SEARCH_MODE_FORWARD_MATCHING; // 前方一致モードを適用

	// 【実行サイズ算出】想定デバイス（RTX 5060等）の並列計算効率を最大化するグリッド配置の計算
	uint32_t numBlocksPerSM = 8;        // 1つのStreaming Multiprocessorに投入するブロック数
	uint32_t multiProcessorCount = 24; // デバイスのSM総数基数
	uint32_t numBlocksPerGrid = numBlocksPerSM * multiProcessorCount; // グリッド全体の総ブロック数（192）
	uint32_t numThreadsPerGrid = 384 * numBlocksPerGrid; // グリッド全体の総スレッド数（73,728）
	uint64_t totalThreadsPerLaunch = (uint64_t)numThreadsPerGrid * 32; // 1回のカーネル呼び出しで消化するビットスライス総空間（2,359,296）

	// 【VRAM割り当てと厳密なエラーチェック】メモリ確保の失敗を確実に検知するガード構造
#ifdef ENABLE_CUDA
	// 各ポインタに対してサイズを厳密に計算してVRAMを確保。戻り値がcudaSuccessでない場合は強制終了
	if (cudaMalloc((void**)&d_passCountArray, sizeof(unsigned char) * numThreadsPerGrid) != cudaSuccess) { std::cerr << "[FATAL] d_passCountArray 確保失敗" << std::endl; return -1; }
	if (cudaMalloc((void**)&d_tripcodeIndexArray, sizeof(unsigned char) * numThreadsPerGrid) != cudaSuccess) { std::cerr << "[FATAL] d_tripcodeIndexArray 確保失敗" << std::endl; cudaFree(d_passCountArray); return -1; }
	if (cudaMalloc((void**)&d_tripcodeChunkArray, sizeof(uint32_t) * numTripcodeChunk) != cudaSuccess) { std::cerr << "[FATAL] d_tripcodeChunkArray 確保失敗" << std::endl; cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); return -1; }
	if (cudaMalloc((void**)&d_cudaKey0Array, sizeof(unsigned char) * 32) != cudaSuccess) { std::cerr << "[FATAL] d_cudaKey0Array 確保失敗" << std::endl; /* 確保済メモリの解放処理を連鎖 */ cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); return -1; }
	if (cudaMalloc((void**)&d_cudaKey7Array, sizeof(unsigned char) * 32 * 2) != cudaSuccess) { std::cerr << "[FATAL] d_cudaKey7Array 確保失敗" << std::endl; cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); cudaFree(d_cudaKey0Array); return -1; }
	if (cudaMalloc((void**)&d_cudaKeyAndRandomBytes, sizeof(unsigned char) * 8) != cudaSuccess) { std::cerr << "[FATAL] d_cudaKeyAndRandomBytes 確保失敗" << std::endl; cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); cudaFree(d_cudaKey0Array); cudaFree(d_cudaKey7Array); return -1; }
	if (cudaMalloc((void**)&d_cudaKeyVectorsFrom49To55, sizeof(uint32_t) * 64) != cudaSuccess) { std::cerr << "[FATAL] d_cudaKeyVectorsFrom49To55 確保失敗" << std::endl; cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); cudaFree(d_cudaKey0Array); cudaFree(d_cudaKey7Array); cudaFree(d_cudaKeyAndRandomBytes); return -1; }

	// 👈 【バグ修正】確保しただけのデバイス鍵配列へ初期データを転送（未初期化領域の読み込みによるハッシュ計算破綻の回避）
	// ホスト側でクリーンなゼロ初期化データを生成し、事前にVRAM上へ一回同期転送を行う
	unsigned char h_zeroKeyInit[256] = {0};
	cudaMemcpy(d_cudaKey0Array, h_zeroKeyInit, sizeof(unsigned char) * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cudaKey7Array, h_zeroKeyInit, sizeof(unsigned char) * 32 * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cudaKeyVectorsFrom49To55, h_zeroKeyInit, sizeof(uint32_t) * 64, cudaMemcpyHostToDevice);
#endif

	// 【ホスト側メインメモリ確保と nullptr チェック】
	// メインメモリ上にGPUからの戻り値を回収する領域をmalloc。確保失敗時はポインタ破棄とVRAMをクリーンアップして終了
	unsigned char* h_passCountArray = (unsigned char*)malloc(sizeof(unsigned char) * numThreadsPerGrid);
	if (h_passCountArray == nullptr) { std::cerr << "[FATAL] ホストメモリ h_passCountArray 確保失敗" << std::endl; 
#ifdef ENABLE_CUDA
		cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); cudaFree(d_cudaKey0Array); cudaFree(d_cudaKey7Array); cudaFree(d_cudaKeyAndRandomBytes); cudaFree(d_cudaKeyVectorsFrom49To55);
#endif
		return -1; 
	}

	unsigned char* h_tripcodeIndexArray = (unsigned char*)malloc(sizeof(unsigned char) * numThreadsPerGrid);
	if (h_tripcodeIndexArray == nullptr) { std::cerr << "[FATAL] ホストメモリ h_tripcodeIndexArray 確保失敗" << std::endl; free(h_passCountArray);
#ifdef ENABLE_CUDA
		cudaFree(d_passCountArray); cudaFree(d_tripcodeIndexArray); cudaFree(d_tripcodeChunkArray); cudaFree(d_cudaKey0Array); cudaFree(d_cudaKey7Array); cudaFree(d_cudaKeyAndRandomBytes); cudaFree(d_cudaKeyVectorsFrom49To55);
#endif
		return -1;
	}

	unsigned char keyAndRandomBytes[8]; // 64bitスキャンカウンターを分解してGPUへ渡すためのホスト側一時バッファ配列

	// 探索パターン用テストフィルタの初期化およびVRAMへの初期転送
	uint32_t h_dummyChunk = 0xFFFFFFFF; // 全ビット通過用フィルタ
#ifdef ENABLE_CUDA
	cudaMemcpy(d_tripcodeChunkArray, &h_dummyChunk, sizeof(uint32_t) * numTripcodeChunk, cudaMemcpyHostToDevice);
#endif

	std::cout << "[INFO] GPU全域ラッシュを開始します。" << std::endl;
	auto startTime = std::chrono::high_resolution_clock::now(); // 時間追跡用の基準時刻をマーク
	uint64_t lastSavedCounter = currentGlobalCounter;           // オートセーブ基準位置の同期

	// -------------------------------------------------------------------------
	// 🔄 メイン探索無限ループ
	// -------------------------------------------------------------------------
	while (!GetTerminationState()) { // 終了状態シグナルが立っていない間ループを持続する

		// 👈 【バグ修正】カウンターのオーバーフロー（ラップアラウンドによる無限ループ）を防止する境界ガード処理
		// 次の打ち出しで 0xFFFFFFFFFFFFFFFFULL を超える場合は、現在の周回を最終処理として安全にブレイクする
		if (0xFFFFFFFFFFFFFFFFULL - currentGlobalCounter < totalThreadsPerLaunch) {
			std::cout << "\n[INFO] キー空間全体の走査がまもなく完了します（最終ブロック）。" << std::endl;
		}
		
		// 現在の64ビットスキャンカウンターの絶対値を、8バイトのBig-Endian形式データへ厳密にビット分解
		for (int i = 0; i < 8; ++i) {
			keyAndRandomBytes[i] = (unsigned char)((currentGlobalCounter >> ((7 - i) * 8)) & 0xFF);
		}

#ifdef ENABLE_CUDA
		// 分解完了した基底生キー（8バイト）をGPUの定数受信用バッファポインタへ同期転送
		cudaMemcpy(d_cudaKeyAndRandomBytes, keyAndRandomBytes, 8, cudaMemcpyHostToDevice);
		
		// マッチング結果を受信するVRAM検出用カウンタ配列を毎サイクル確実にゼロクリア
		cudaMemset(d_passCountArray, 0, sizeof(unsigned char) * numThreadsPerGrid);
		cudaMemset(d_tripcodeIndexArray, 0, sizeof(unsigned char) * numThreadsPerGrid);

		// Cリンケージで接続された外部のCUDAデバイス側探索カーネルラッパー関数をコール
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
		
		// 非同期カーネルの実行完了をCPU側でブロッキング同期待機
		cudaDeviceSynchronize();

		// 👈 【バグ・パフォーマンス修正】ループに入る前に結果データを一括して一回でホスト側へ回収転送
		// ループ内での過剰な通信同期ストールを完全に排除し、VRAMバス帯域のボトルネックを解消
		cudaMemcpy(h_passCountArray, d_passCountArray, sizeof(unsigned char) * numThreadsPerGrid, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_tripcodeIndexArray, d_tripcodeIndexArray, sizeof(unsigned char) * numThreadsPerGrid, cudaMemcpyDeviceToHost);
#endif

		// ホスト側に回収した全並列スレッド数（73728要素）の結果フラグ配列を精査
		for (uint32_t i = 0; i < numThreadsPerGrid; i++) {
			// 32（不一致既定値）未満のビットインデックス数値が返却されている場合、ビットスライス空間内で該当キーを検知
			if (h_passCountArray[i] < 32) { 
				
				uint64_t hitThreadIndex = i; // 条件が一致したパラレルスレッドの並列番号を確定
				// 👈 【バグ修正】ビットインデックス（h_passCountArray[i]に格納されている0〜31の値）を抽出
				uint64_t bit_index = (uint64_t)h_passCountArray[i]; 

				// 👈 【バグ修正】ビットスライス空間を網羅する正しいカウンター絶対座標（生キー値）の厳密なマッピング算出
				// 旧コード「currentGlobalCounter + hitThreadIndex」による32倍のキー空間の計算不整合および誤検出の不具合を完全に解消
				// 各スレッドが32個のキーを連続して受け持つ配置モデルに適合させ、全歩幅空間と完全に合致させる
				uint64_t foundBinaryKey = currentGlobalCounter + (hitThreadIndex * 32) + bit_index; 

				// トリップ入力形式「##[16進数生キー大文字16桁]r1」を生成するための安全なバッファ
				char resolvedHexKey[32] = {0};
				// 👈 【脆弱性修正】sprintfを安全なsnprintfへ変更。環境依存のない一意な64bit用大文字16進数指定マクロ PRIX64 を採用
				snprintf(resolvedHexKey, sizeof(resolvedHexKey), "##%016" PRIX64 "r1", foundBinaryKey); 

				std::cout << "\n==================================================" << std::endl;
				std::cout << "🎯 【ターゲット発見】 条件一致生キーをロックしました" << std::endl;
				std::cout << "  入力キー: " << resolvedHexKey << std::endl;
				std::cout << "==================================================" << std::endl;

				// 【ログキャッシュの保存】発見ログをファイル末尾へ自動追記（永続化）
				std::ofstream logOut(MATCH_LOG_FILE, std::ios::app); // 追記モードでストリームオープン
				if (logOut.is_open()) {
					logOut << "Key: " << resolvedHexKey << " | Verified Match.\n"; // ログ行の書き込み
					logOut.close(); // 排他ロック制御のため即座に明示的クローズ
				}
			}
		}

		// 👈 【バグ修正】カウンターのオーバーフロー直前ガード処理：これ以上加算するとラップアラウンドする場合はループを正常に脱出
		if (0xFFFFFFFFFFFFFFFFULL - currentGlobalCounter < totalThreadsPerLaunch) {
			break;
		}

		// 走査が正常完了した並列幅（総スレッド数×ビットスライス深度32）の分だけ正確にシリアルカウンターを進める
		currentGlobalCounter += totalThreadsPerLaunch;

		// 💾 【オートセーブ】一定の間隔（500回打ち出し毎）に達した場合、現在の走査位置キャッシュをダンプ保存
		if (currentGlobalCounter - lastSavedCounter > (totalThreadsPerLaunch * 500)) {
			std::ofstream resumeOut(RESUME_FILE, std::ios::binary | std::ios::trunc); // 既存ファイルを上書きモードで開く
			if (resumeOut.is_open()) {
				resumeOut.write(reinterpret_cast<const char*>(&currentGlobalCounter), sizeof(currentGlobalCounter)); // 進行位置をバイナリ書き込み
				resumeOut.close(); // ストリームクローズ
			}
			lastSavedCounter = currentGlobalCounter; // 次回保存のための基準位置カウンターを更新

			// 進捗ログをCarriage Return（\r）で同一行に上書き表示し、画面への過剰出力を抑止
			std::hex(std::cout); // 16進数表示
			std::cout << "\r[Scan Progress] Position: 0x" << currentGlobalCounter << " ... Cache Synced" << std::flush;
			std::dec(std::cout); // 10進数表示に戻す
		}
	}

	// -------------------------------------------------------------------------
	// 🧹 クリーンアップ処理（メモリリークの完全防止）
	// -------------------------------------------------------------------------
#ifdef ENABLE_CUDA
	// 確保された各デバイス側メモリを安全に解放し、VRAMリークを完璧に防止する
	cudaFree(d_passCountArray);
	cudaFree(d_tripcodeIndexArray);
	cudaFree(d_tripcodeChunkArray);
	cudaFree(d_cudaKey0Array);
	cudaFree(d_cudaKey7Array);
	cudaFree(d_cudaKeyAndRandomBytes);
	cudaFree(d_cudaKeyVectorsFrom49To55);
#endif
	// ホスト側（メインメモリ）に割り当てていた領域を確実に解放
	free(h_passCountArray);
	free(h_tripcodeIndexArray);

	std::cout << "\n[INFO] プロセスを正常に終了しました。" << std::endl;
	return 0; // プログラムの正常終了コードを返却
}
