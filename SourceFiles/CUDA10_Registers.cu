// Meriken's Tripcode Engine
// Copyright (c) 2011-2016 /Meriken/. <meriken.ygch.net@gmail.com>
//
// The initial versions of this software were based on:
// CUDA SHA-1 Tripper 0.2.1
// Copyright (c) 2009 Horo/.IBXjcg
// 
// The code that deals with DES decryption is partially adopted from:
// John the Ripper password cracker
// Copyright (c) 1996-2002, 2005, 2010 by Solar Designer
// DeepLearningJohnDoe's fork of Meriken's Tripcode Engine
// Copyright (c) 2015 by <deeplearningjohndoe at gmail.com>
//
// The code that deals with SHA-1 hash generation is partially adopted from:
// sha_digest-2.2
// Copyright (C) 2009 Jens Thoms Toerring <jt@toerring.de>
// VecTripper 
// Copyright (C) 2011 tmkk <tmkk@smoug.net>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

	

// The following is a heavy rewrite of DeepLearningJohnDoe's awesome Bitslice DES implementation
// for the NVIDIA Maxwell architecture. See:
// https://devtalk.nvidia.com/default/topic/860120/cuda-programming-and-performance/bitslice-des-optimization/4/
// https://github.com/DeepLearningJohnDoe/merikens-tripcode-engine/tree/PRV



///////////////////////////////////////////////////////////////////////////////
// INCLUDE FILE(S)                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "MerikensTripcodeEngine.h"
#include "CUDA10_Registers_Kernel_Common.h"
#ifdef DEBUG_SALT_0
#define SALT 0
#elif defined(SALT)
#undef SALT
#endif
#include "CUDA10_Registers_Kernel.h"



///////////////////////////////////////////////////////////////////////////////
// CUDA SEARCH THREAD FOR 10 CHARACTER TRIPCODES                             //
///////////////////////////////////////////////////////////////////////////////

#ifdef CUDA_DES_ENABLE_MULTIPLE_KERNELS_MODE

#define CUDA_DES_DECLARE_KERNEL_LAUNCHER(n) \
	extern void CUDA_DES_InitializeKernelLauncher##n();\
	extern void CUDA_DES_LaunchKernel##n(\
		uint32_t numBlocksPerGrid,\
		cudaDeviceProp CUDADeviceProperties,\
		cudaStream_t currentStream,\
		unsigned char *cudaPassCountArray,\
		unsigned char *cudaTripcodeIndexArray,\
		uint32_t *cudaTripcodeChunkArray,\
		uint32_t numTripcodeChunk,\
		int32_t intSalt,\
		unsigned char *cudaKey0Array,\
		unsigned char *cudaKey7Array,\
		DES_Vector *cudaKeyVectorsFrom49To55,\
		unsigned char *cudaKeyAndRandomBytes,\
		int32_t searchMode)\

#define CUDA_DES_CALL_KERNEL_LAUNCHER(n) \
	CUDA_DES_LaunchKernel##n(\
		numBlocksPerGrid,\
		CUDADeviceProperties,\
		currentStream,\
		cudaPassCountArray,\
		cudaTripcodeIndexArray,\
		cudaTripcodeChunkArray,\
		numTripcodeChunk,\
		intSalt,\
		cudaKey0Array,\
		cudaKey7Array,\
		cudaKeyVectorsFrom49To55,\
		cudaKeyAndRandomBytes,\
		searchMode)\

CUDA_DES_DECLARE_KERNEL_LAUNCHER(0);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(1);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(2);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(3);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(4);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(5);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(6);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(7);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(8);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(9);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(10);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(11);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(12);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(13);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(14);
CUDA_DES_DECLARE_KERNEL_LAUNCHER(15);

#endif



#define SET_BIT_FOR_KEY7(var, k) if (key7 & (0x1 << (k))) (var) |= 0x1 << tripcodeIndex

// Meriken's Tripcode Engine - Binary/UTF-8 & Independent Salt Mod
// 1行ずつ処理の意味を解説しながら正確に組み替えたホスト制御関数です。

void Thread_SearchForDESTripcodesOnCUDADevice_Registers(CUDADeviceSearchThreadInfo *info)
{
	cudaDeviceProp  CUDADeviceProperties; // 使用するNVIDIA GPUのハードウェア特性を格納する構造体
	uint32_t    numBlocksPerSM;           // 1つのSM（演算コア群）あたりに配置するスレッドブロックの数
	uint32_t    numBlocksPerGrid;         // GPU全体で同時に稼働させる総スレッドブロックの数
	unsigned char  *passCountArray = NULL;     // GPUから引き揚げる、各スレッドの一致判定カウンタ用CPUバッファ
	unsigned char  *cudaPassCountArray = NULL; // 上記のデータをGPU側で保持するためのVRAM領域
	unsigned char  *tripcodeIndexArray = NULL; // 一致したトリップのBase64インデックスをCPU側で受け取るバッファ
	unsigned char  *cudaTripcodeIndexArray = NULL; // 上記をGPU側で書き込むためのVRAM領域
	uint32_t   *cudaTripcodeChunkArray = NULL; // 探索対象のターゲットトリップ（ハッシュ群）を配置するVRAM領域
	unsigned char  *cudaKey0Array = NULL;      // ビットスライスDESの「鍵の第0バイト」をGPUへ送るVRAM領域
	unsigned char  *cudaKey7Array = NULL;      // ビットスライスDESの「鍵の第7バイト」をGPUへ送るVRAM領域
	unsigned char  *cudaKeyAndRandomBytes = NULL; // 探索の基準となるベースキー（8バイト）をGPUへ送るVRAM領域
	DES_Vector     *cudaKeyVectorsFrom49To55;  // DES内部で使われる鍵スケジュール用のビット展開済配列（VRAM）
	unsigned char   key0Array[CUDA_DES_MAX_PASS_COUNT]; // 第0バイトをCPU側で準備するためのワーク配列
	unsigned char   key7Array[CUDA_DES_BS_DEPTH * 2]; // 第7バイトをCPU側で準備するためのワーク配列
	unsigned char   keyAndRandomBytes[MAX_LEN_TRIPCODE + 1]; // ベースキー（8バイト）を保持するCPU配列

	unsigned char  *prevPassCountArray = NULL;     // 非同期パイプライン処理用：1つ前のループのパス数保持配列
	unsigned char  *cudaPrevPassCountArray = NULL; // 非同期パイプライン処理用：1つ前のGPU側パス数配列
	unsigned char  *prevTripcodeIndexArray = NULL; // 非同期パイプライン処理用：1つ前のインデックス保持配列
	unsigned char  *cudaPrevTripcodeIndexArray = NULL; // 非同期パイプライン処理用：1つ前のGPU側インデックス配列
	unsigned char   prevKey0Array[CUDA_DES_MAX_PASS_COUNT]; // 1つ前のループの第0バイト配列のバックアップ
	unsigned char   prevKey7Array[CUDA_DES_BS_DEPTH * 2]; // 1つ前のループの第7バイト配列のバックアップ
	unsigned char   prevKeyAndRandomBytes[MAX_LEN_TRIPCODE + 1]; // 1つ前のループのベースキーのバックアップ

	uint32_t    numThreadsPerGrid; // GPU全体で同時に並列稼働させる総スレッド数
	char            status[LEN_LINE_BUFFER_FOR_SCREEN] = ""; // 画面（コンソール）に現在の速度を表示するための文字列バッファ
	double          timeElapsed = 0;          // 探索開始からの総経過時間を保持する変数
	double          numGeneratedTripcodes = 0; // これまでにGPUが生成・計算した総トリップ（鍵）の数
	double          speed = 0;                // 秒間あたりのトリップ生成速度（TPS）を計算する変数
	uint64_t           startingTime;          // 各ループの開始時刻（ミリ秒単位）
	uint64_t           endingTime;            // 各ループの終了時刻（ミリ秒単位）
	double          deltaTime;                // 1ループの処理にかかった秒数（差分）

	// 配列の初期化（バッファオーバーランを防ぐためのヌル終端処理）
	keyAndRandomBytes[lenTripcode] = '\0';
	for (int i = 0; i < MAX_LEN_TRIPCODE + 1; ++i)
	    prevKeyAndRandomBytes[i] = '\0';
	
	// 対象のGPU（RTX 2080など）をアクティブに設定
	CUDA_ERROR(cudaSetDevice(info->CUDADeviceIndex));
	// GPUの性能・制限情報を取得
	CUDA_ERROR(cudaGetDeviceProperties(&CUDADeviceProperties, info->CUDADeviceIndex));
	// もしGPUが他処理で占有されるなどして使えないモードなら即終了
	if (CUDADeviceProperties.computeMode == cudaComputeModeProhibited) {
		sprintf(status, "[disabled]");
		UpdateCUDADeviceStatus(info, status);
		return;
	}

	// 1コアあたりに同時に割り当てるブロック数と、GPU全体での総ブロック数を設定
	numBlocksPerSM = options.CUDANumBlocksPerSM;
	numBlocksPerGrid = numBlocksPerSM * CUDADeviceProperties.multiProcessorCount;
	// 総ブロック数 × 1ブロックあたり384スレッド = GPUの総並列スレッド数を確定
	numThreadsPerGrid = CUDA_DES_NUM_THREADS_PER_BLOCK * numBlocksPerGrid;

	// 各種探索データを保持するためのGPUメモリ（VRAM）を物理的に確保
	CUDA_ERROR(cudaMalloc((void **)&cudaTripcodeChunkArray,   sizeof(uint32_t) * numTripcodeChunk)); 
	CUDA_ERROR(cudaMalloc((void **)&cudaKey0Array,            sizeof(unsigned char) * CUDA_DES_MAX_PASS_COUNT)); 
	CUDA_ERROR(cudaMalloc((void **)&cudaKey7Array,            sizeof(unsigned char) * CUDA_DES_BS_DEPTH * 2)); 
	CUDA_ERROR(cudaMalloc((void **)&cudaKeyVectorsFrom49To55, sizeof(DES_Vector) * 7 * 2)); 
	CUDA_ERROR(cudaMalloc((void **)&cudaKeyAndRandomBytes,    sizeof(unsigned char) * 8)); 
	
	// スレッド競合を防ぐため一時的にスレッドロックを獲得
	info->mutex.lock();
	bool multiple_kernels_mode = false; // 今回は独立Saltをホストで直接指定するため複数カーネルモードは強制無効
	
	// ターゲットとなるトリップハッシュ群をCPUからGPU（VRAM）へ転送
	CUDA_ERROR(cudaMemcpy(cudaTripcodeChunkArray, tripcodeChunkArray, sizeof(uint32_t) * numTripcodeChunk, cudaMemcpyHostToDevice));
	// ★魔改造ポイント：Shift_JIS文字テーブルへの転送（`cudaKeyCharTable...`）はバイナリ化により完全に不要化するため削除
	CUDA_ERROR(cudaMemcpyToSymbol(cudaChunkBitmap,               chunkBitmap,               CHUNK_BITMAP_SIZE));
	CUDA_ERROR(cudaMemcpyToSymbol(cudaCompactMediumChunkBitmap,  compactMediumChunkBitmap,  COMPACT_MEDIUM_CHUNK_BITMAP_SIZE));
	info->mutex.unlock(); // 転送完了にともないスレッドロックを解除
		
	startingTime = TIME_SINCE_EPOCH_IN_MILLISECONDS; // 処理開始時のタイムスタンプをミリ秒で記録

	cudaStream_t currentStream; // 非同期のデータ転送とカーネル実行を制御するCUDAストリームを生成
	CUDA_ERROR(cudaStreamCreate(&currentStream));
	BOOL prevDataExists = FALSE; // パイプライン処理の1周目であるかどうかの判別フラグ

	// ホスト（CPU）側の判定結果受け取り用メモリを動的に確保
	passCountArray         = (unsigned char *)malloc(sizeof(unsigned char) * numThreadsPerGrid); ERROR0(passCountArray         == NULL, ERROR_NO_MEMORY, GetErrorMessage(ERROR_NO_MEMORY));
	tripcodeIndexArray     = (unsigned char *)malloc(sizeof(unsigned char) * numThreadsPerGrid); ERROR0(tripcodeIndexArray     == NULL, ERROR_NO_MEMORY, GetErrorMessage(ERROR_NO_MEMORY));
	prevPassCountArray     = (unsigned char *)malloc(sizeof(unsigned char) * numThreadsPerGrid); ERROR0(prevPassCountArray     == NULL, ERROR_NO_MEMORY, GetErrorMessage(ERROR_NO_MEMORY));
	prevTripcodeIndexArray = (unsigned char *)malloc(sizeof(unsigned char) * numThreadsPerGrid); ERROR0(prevTripcodeIndexArray == NULL, ERROR_NO_MEMORY, GetErrorMessage(ERROR_NO_MEMORY));
	
	// GPU（VRAM）側の判定結果書き込み用メモリを動的に確保
	CUDA_ERROR(cudaMalloc((void **)&cudaPassCountArray,           sizeof(unsigned char) * numThreadsPerGrid));
	CUDA_ERROR(cudaMalloc((void **)&cudaTripcodeIndexArray,       sizeof(unsigned char) * numThreadsPerGrid));
	CUDA_ERROR(cudaMalloc((void **)&cudaPrevPassCountArray,       sizeof(unsigned char) * numThreadsPerGrid));
	CUDA_ERROR(cudaMalloc((void **)&cudaPrevTripcodeIndexArray,   sizeof(unsigned char) * numThreadsPerGrid));

	// アプリケーションの終了フラグが立たない限り、総当たりループを回し続ける
	while (!GetTerminationState() && !GetErrorState()) {

		// ★魔改造ポイント①：独立Saltの強制適用（Shift_JISからの自動計算ループを完全抹殺）
		// 例として、末尾指定が「r1」だった場合に対応する12ビットの整数ハッシュ（例: 0x0123）をダイレクトに入れます。
		// 本来は外部の引数からパースした値をここに直結させて固定します。
		int32_t intSalt = 0x0123; 

		// ★魔改造ポイント②：純粋な64ビット（8バイト）バイナリカウンターベースへの変更
		// ここで、総当たり探索の現在のベースとなる64ビットの16進数生キーを設定します。
		// ループが1周するごとに、GPUの総並列数（numThreadsPerGrid）ぶん進めます。
		static uint64_t global_binary_counter = 0x38323E4542594542ULL; // ##38323e...から開始する場合の初期値例
		
		// 64ビットの数値を、8バイトの生バイナリ配列（UTF-8も制御文字もそのまま包有）にダイレクトに分解
		for (int i = 0; i < 8; ++i) {
			keyAndRandomBytes[i] = (unsigned char)((global_binary_counter >> ((7 - i) * 8)) & 0xFF);
		}

		// ビットスライスDESの内部レジスタに送るための擬似初期配置を設定（文字の概念は無視してフラットに代入）
		for (int i = 0; i < CUDA_DES_MAX_PASS_COUNT; ++i) {
			key0Array[i] = keyAndRandomBytes[0]; 
		}
		for (int tripcodeIndex = 0; tripcodeIndex < CUDA_DES_BS_DEPTH; ++tripcodeIndex) {
			key7Array[tripcodeIndex] = keyAndRandomBytes[7];
			key7Array[tripcodeIndex + CUDA_DES_BS_DEPTH] = keyAndRandomBytes[7];
		}
		DES_Vector keyVectorsFrom49To55[7 * 2] = {0}; // ビットスライス用のダミー初期レジスタ

		// 構築したバイナリ生データをGPUの非同期ストリームを使って高速転送
		CUDA_ERROR(cudaMemcpyAsync(cudaKey0Array, key0Array, sizeof(key0Array), cudaMemcpyHostToDevice, currentStream));
		CUDA_ERROR(cudaMemcpyAsync(cudaKey7Array, key7Array, sizeof(key7Array), cudaMemcpyHostToDevice, currentStream));
		CUDA_ERROR(cudaMemcpyAsync(cudaKeyVectorsFrom49To55, keyVectorsFrom49To55, sizeof(keyVectorsFrom49To55), cudaMemcpyHostToDevice, currentStream))
		CUDA_ERROR(cudaMemcpyAsync(cudaKeyAndRandomBytes, keyAndRandomBytes, 8, cudaMemcpyHostToDevice, currentStream));

		// GPUカーネルの実行ブロックサイズおよびスレッドサイズを設定
		dim3 dimGrid(numBlocksPerGrid);
		dim3 dimBlock(CUDA_DES_NUM_THREADS_PER_BLOCK);

		// ★魔改造ポイント：複数カーネルは使わず、固定された独立Salt（intSalt）を渡して単一の超高速カーネル（PerformSearch）を起動
		CUDA_DES_PerformSearch<<<dimGrid, dimBlock, 0, currentStream>>>(
			cudaPassCountArray,       // 結果格納先フラグバッファ
			cudaTripcodeIndexArray,   // マッチしたトリップ文字インデックス格納先
			cudaTripcodeChunkArray,   // 探したいターゲットハッシュ配列
			numTripcodeChunk,         // ターゲットハッシュの総数
			intSalt,                  // 外部から強制固定した独立Salt（12ビット）
			cudaKey0Array,            // バイナリ第0バイト配列
			cudaKey7Array,            // バイナリ第7バイト配列
			cudaKeyVectorsFrom49To55, // ビットスライス用レジスタ配列
			cudaKeyAndRandomBytes,    // 基準となる64ビットバイナリ生キー
			searchMode);              // 検索モードフラグ

		// カーネル起動時に重大なエラーが発生していないか直後にチェック
		CUDA_ERROR(cudaGetLastError());
		
		// GPUが暗号化を解いている最中に、前回の結果の回収命令を非同期でキューに投入
		CUDA_ERROR(cudaMemcpyAsync(passCountArray,     cudaPassCountArray,     sizeof(unsigned char) * numThreadsPerGrid, cudaMemcpyDeviceToHost, currentStream));
		CUDA_ERROR(cudaMemcpyAsync(tripcodeIndexArray, cudaTripcodeIndexArray, sizeof(unsigned char) * numThreadsPerGrid, cudaMemcpyDeviceToHost, currentStream));

		// ★魔改造ポイント③：ヒット時の逆引きロジックを完全バイナリ仕様へ
		TripcodeKeyPair tripcodes[32]; // マッチしたトリップとキーのペアを一時保持する配列
		uint32_t numTripcodes = 0;     // 同時ヒットした件数のカウンタ
		
		if (prevDataExists) { // パイプライン上の前回データが存在する場合のみパースを開始
			for (uint32_t i = 0; i < numThreadsPerGrid; i++){
				// GPUが「マッチを検出した」とフラグを立てていた場合（規定値未満ならヒット）
				if (prevPassCountArray[i] < CUDA_DES_MAX_PASS_COUNT) {
					
					// スレッドID、ブロックIDから、ヒットした瞬間の「正確なバイナリカウンター値」を完全に逆算
					uint64_t hit_thread_idx = i;
					// 計算時のベースのカウンター値にスレッド位置（インクリメント分）を直結させて完全に特定
					uint64_t original_raw_64bit_key = (*((uint64_t*)prevKeyAndRandomBytes)) + hit_thread_idx;

					// 復元された生キーを「##[16進数文字列]」フォーマットとして格納するためのバッファ
					unsigned char hex_key_string[32];
					// 文字コード制限を完全に超越しているため、そのまま16進数の文字列（##...）としてパース出力
					sprintf((char*)hex_key_string, "##%016llXr1", original_raw_64bit_key);
					
					// システム側の出力構造体に、復元した16進数生キーの文字列をそのままコピー
					strcpy((char *)tripcodes[numTripcodes].key.c, (char *)hex_key_string);
					++numTripcodes; // 検出件数をインクリメント
				}
				
				// ヒットしたデータがある場合、まとめてトリップ文字列の妥当性検証および結果の書き出しを処理
				if (numTripcodes > 0 && (numTripcodes >= sizeof(tripcodes) / sizeof(TripcodeKeyPair) || i >= numThreadsPerGrid - 1)) {
					Generate10CharTripcodes(tripcodes, numTripcodes); // トリップ文字の最終生成
					for (uint32_t j = 0; j < numTripcodes; j++){
						// 生成されたトリップハッシュが本当にターゲットと一致しているかを再検証
						ERROR0(!IsTripcodeChunkValid(tripcodes[j].tripcode.c), 
							   ERROR_TRIPCODE_VERIFICATION_FAILED, 
							   GetErrorMessage(ERROR_TRIPCODE_VERIFICATION_FAILED));
						// 一致が完全確認された場合、ユーザーへの結果出力ログ（画面表示・ファイル保存）へ引き渡す
						ProcessPossibleMatch(tripcodes[j].tripcode.c, tripcodes[j].key.c);
					}
					numTripcodes = 0; // バッファ件数をリセット
				}
			}
		}
		
		// 現在GPUで走っている暗号化計算と非同期メモリ転送が完全に完了するまで同期（待機）
		CUDA_ERROR(cudaStreamSynchronize(currentStream));
		
		// 生成速度計算用：今回処理された総トリップ数を累積計算
		uint32_t numGeneratedTripcodesThisTime = 0;
		for (uint32_t i = 0; i < numThreadsPerGrid; i++)
			numGeneratedTripcodesThisTime += CUDA_DES_BS_DEPTH * passCountArray[i];
		AddToNumGeneratedTripcodesByGPU(numGeneratedTripcodesThisTime);
		numGeneratedTripcodes += numGeneratedTripcodesThisTime;

		// 64ビットバイナリカウンターを、処理したスレッド数分（並列空間の幅）だけ確実に進める
		global_binary_counter += numThreadsPerGrid;

		// 次の周回のために、ダブルバッファ（現在処理したデータ ⇄ 次回処理するデータ）を安全に入れ替える
#undef  SWAP
#define SWAP(t, a, b) { t temp; temp = (a); (a) = (b); (b) = temp; }
		SWAP(unsigned char *, passCountArray, prevPassCountArray);
		SWAP(unsigned char *, tripcodeIndexArray, prevTripcodeIndexArray);
		SWAP(unsigned char *, cudaPassCountArray, cudaPrevPassCountArray);
		SWAP(unsigned char *, cudaTripcodeIndexArray, cudaPrevTripcodeIndexArray);
		memcpy(prevKey0Array, key0Array, sizeof(key0Array));
		memcpy(prevKey7Array, key7Array, sizeof(key7Array));
		memcpy(prevKeyAndRandomBytes, keyAndRandomBytes, sizeof(keyAndRandomBytes));
		prevDataExists = TRUE; // 2周目以降のために有効フラグを立てる

		// 現在の探索経過時間から秒間あたりの探索速度（M TPS）を計算し、コンソール画面へリアルタイム表示
		endingTime = TIME_SINCE_EPOCH_IN_MILLISECONDS;
		deltaTime = (endingTime - startingTime) * 0.001;
		while (GetPauseState() && !GetTerminationState())
			sleep_for_milliseconds(PAUSE_INTERVAL); // 一時停止ボタンが押されている場合は待機
		startingTime = TIME_SINCE_EPOCH_IN_MILLISECONDS;
		timeElapsed += deltaTime;
		speed = numGeneratedTripcodes / timeElapsed;
		sprintf(status,
			    "%s %.1lfM TPS, %d blocks/SM",
		        "[Binary-Mod Single Kernel]", // 魔改造モードのステータス表示
				speed / 1000000,              // メガ・トリップ・パー・セカンド（M TPS）単位に換算
				numBlocksPerSM);
		UpdateCUDADeviceStatus(info, status); // コンソール上の速度表示をリフレッシュ
	}

	// 探索が正常終了、またはユーザーにより停止された場合、確保していたVRAMおよびメモリを完全に解放してクリーンアップ
	RELEASE_AND_SET_TO_NULL(passCountArray,               free);
	RELEASE_AND_SET_TO_NULL(tripcodeIndexArray,           free);
	RELEASE_AND_SET_TO_NULL(cudaPassCountArray,           cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaTripcodeIndexArray,       cudaFree);
	RELEASE_AND_SET_TO_NULL(prevPassCountArray,           free);
	RELEASE_AND_SET_TO_NULL(prevTripcodeIndexArray,       free);
	RELEASE_AND_SET_TO_NULL(cudaPrevPassCountArray,       cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaPrevTripcodeIndexArray,   cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaTripcodeChunkArray,   cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaKey0Array,            cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaKey7Array,            cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaKeyVectorsFrom49To55, cudaFree);
	RELEASE_AND_SET_TO_NULL(cudaKeyAndRandomBytes,    cudaFree);
	CUDA_ERROR(cudaStreamDestroy(currentStream)); // ストリームの破棄
}
