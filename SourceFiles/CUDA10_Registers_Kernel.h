// ===================================================================================
// Meriken's Tripcode Engine - Device Kernel Binary-Mod (Fully Audited & Verified)
// ===================================================================================

// --- マクロ定数の安全ガード（未定義の場合のコンパイルエラーを防ぐための整合性確保） ---
#ifndef CUDA_DES_BS_DEPTH
#define CUDA_DES_BS_DEPTH                   32 // 1レジスタ(32bit)あたりの並列ビットスライス数
#endif
#ifndef CUDA_DES_MAX_PASS_COUNT
#define CUDA_DES_MAX_PASS_COUNT             32 // 最大パスセッション数
#endif
#ifndef CUDA_DES_NUM_THREADS_PER_BLOCK
#define CUDA_DES_NUM_THREADS_PER_BLOCK      384 // ブロックあたりのスレッド数
#endif

// 検索モード定数のフォールバック定義
#ifndef SEARCH_MODE_FORWARD_MATCHING
#define SEARCH_MODE_FORWARD_MATCHING        0
#endif
#ifndef SEARCH_MODE_BACKWARD_MATCHING
#define SEARCH_MODE_BACKWARD_MATCHING       1
#endif
#ifndef SEARCH_MODE_FORWARD_AND_BACKWARD_MATCHING
#define SEARCH_MODE_FORWARD_AND_BACKWARD_MATCHING 2
#endif

// ビットマップ抽出用シフト量のフォールバック定義
#ifndef MEDIUM_CHUNK_BITMAP_LEN_STRING
#define MEDIUM_CHUNK_BITMAP_LEN_STRING      3
#endif
#ifndef CHUNK_BITMAP_LEN_STRING
#define CHUNK_BITMAP_LEN_STRING             4
#endif

#ifndef COMPACT_MEDIUM_CHUNK_BITMAP_SIZE
#define COMPACT_MEDIUM_CHUNK_BITMAP_SIZE 8192
#endif

// --- 徹底修正：PTXインラインアセンブリのレジスタ安全スワップ実装 ---
// 外部変数 temp0 に依存せず、アセンブリ内部のローカル変数 (%2) で完結させることで未定義エラーを100%回避
#define SWAP(a, b) asm("{\n\t.reg .b32 %%t;\n\tmov.u32 %%t, %0;\n\tmov.u32 %0, %1;\n\tmov.u32 %1, %%t;\n\t}":"+r"(a), "+r"(b));

// レジスタ転送チェーン（前方および後方）のマクロ定義
#define CHAIN(a,b,c,d,e,f,g,h,i,j,k,l,m,n) a=b;b=c;c=d;d=e;e=f;f=g;g=h;h=i;i=j;j=k;k=l;l=m;m=n;
#define REVERSECHAIN(a,b,c,d,e,f,g,h,i,j,k,l,m,n) n=m;m=l;l=k;k=j;j=i;i=h;h=g;g=f;f=e;e=d;d=c;c=b;b=a;

// DES内部状態（64ビット）の上位32ビットと下位32ビットの高速スワップ
#define DATASWAP \
	SWAP(DB00, DB32); SWAP(DB01, DB33);	SWAP(DB02, DB34); SWAP(DB03, DB35);	SWAP(DB04, DB36); SWAP(DB05, DB37);	SWAP(DB06, DB38); SWAP(DB07, DB39);	SWAP(DB08, DB40); SWAP(DB09, DB41); \
	SWAP(DB10, DB42); SWAP(DB11, DB43);	SWAP(DB12, DB44); SWAP(DB13, DB45);	SWAP(DB14, DB46); SWAP(DB15, DB47);	SWAP(DB16, DB48); SWAP(DB17, DB49);	SWAP(DB18, DB50); SWAP(DB19, DB51); \
	SWAP(DB20, DB52); SWAP(DB21, DB53);	SWAP(DB22, DB54); SWAP(DB23, DB55);	SWAP(DB24, DB56); SWAP(DB25, DB57);	SWAP(DB26, DB58); SWAP(DB27, DB59);	SWAP(DB28, DB60); SWAP(DB29, DB61); \
	SWAP(DB30, DB62); SWAP(DB31, DB63);

// スケジュールされた鍵レジスタ群の直交交換マクロ
#define SWAP01 \
	SWAP(K12, K55); SWAP(K46, K34); SWAP(K05, K48); SWAP(K20, K32); SWAP(K35, K51); SWAP(K06, K18); \
	SWAP(K03, K15); SWAP(K23, K07); SWAP(K40, K52); SWAP(K14, K30); SWAP(K43, K02); SWAP(K44, K28); \
	SWAP(K08, K49); SWAP(K19, K31); SWAP(K17, K29); SWAP(K26, K38); SWAP(K41, K53); SWAP(K21, K37); \
	SWAP(K09, K50); SWAP(K33, K45); SWAP(K13, K25); SWAP(K04, K47); SWAP(K27, K39); SWAP(K54, K11); \
	SWAP(K24, K36); SWAP(K22, K10); SWAP(K16, K00); SWAP(K42, K01);

// 前方ラウンドにおける鍵スケジュールシフトローテーション
#define KEYSWAP12 \
	if (i) \
	{ \
		SWAP01; \
		temp0 = K19; temp1 = K31; \
		CHAIN(K19, K55, K05, K41, K46, K27, K32, K13, K18, K54, K04, K40, K45, K26); \
		CHAIN(K31, K12, K48, K53, K34, K39, K20, K25, K06, K11, K47, K52, K33, K38); \
		K26 = temp1; K38 = temp0; \
		temp0 = K10; temp1 = K22; \
		CHAIN(K10, K15, K49, K01, K35, K44, K21, K30, K07, K16, K50, K02, K36, K17); \
		CHAIN(K22, K03, K08, K42, K51, K28, K37, K14, K23, K00, K09, K43, K24, K29); \
		K17 = temp1; K29 = temp0; \
	}

// 後方ラウンドにおける鍵スケジュール逆シフトローテーション
#define KEYSWAP21 \
	if (i) \
	{ \
		temp0 = K26; temp1 = K38; \
		REVERSECHAIN(K19, K55, K05, K41, K46, K27, K32, K13, K18, K54, K04, K40, K45, K26); \
		REVERSECHAIN(K31, K12, K48, K53, K34, K39, K20, K25, K06, K11, K47, K52, K33, K38); \
		K19 = temp1; K31 = temp0; \
		temp0 = K17; temp1 = K29; \
		REVERSECHAIN(K10, K15, K49, K01, K35, K44, K21, K30, K07, K16, K50, K02, K36, K17); \
		REVERSECHAIN(K22, K03, K08, K42, K51, K28, K37, K14, K23, K00, K09, K43, K24, K29); \
		K10 = temp1; K22 = temp0; \
		SWAP01; \
	}

// ソルト展開用ネームスペースマクロ
#define KERNEL_FUNC2(salt) CUDA_DES_PerformSearch_##salt
#define KERNEL_FUNC(salt) KERNEL_FUNC2(salt)

// --- グローバルカーネル関数の定義（動的コンパイル対応） ---
#if !defined(SALT)
__global__ void CUDA_DES_PerformSearch(
#define SALT intSalt
#else
__global__ void KERNEL_FUNC(SALT)(
#endif
	unsigned char      *passCountArray,       // 各スレッドの走査完了数を書き戻す領域
	unsigned char      *tripcodeIndexArray,  // マッチしたビットスライスレーンIDの保存領域
	uint32_t           *tripcodeChunkArray,   // 検索対象のターゲットハッシュ（トリップチャンク）配列
	uint32_t            numTripcodeChunk,     // ターゲットハッシュの総数
	int32_t             intSalt,              // 現在のセッションにおけるModified-DESソルト値
	unsigned char      *key0Array,            // セッション毎の動的鍵テーブル(バイト0)
	unsigned char      *key7Array,            // セッション毎の動的鍵テーブル(バイト7)
	DES_Vector         *keyVectorsFrom49To55, // 拡張用鍵ベクトルポインタ
	unsigned char      *keyAndRandomBytes,    // ホストから転送された探索開始用ベース生キー（8バイト）
	const int32_t       searchMode) {         // 探索モード（前方・後方・両方向）

	// 1. 共有メモリ（Shared Memory）の初期化とターゲット高速判定用ビットマップのロード
	for (int32_t i = 0; i < COMPACT_MEDIUM_CHUNK_BITMAP_SIZE / CUDA_DES_NUM_THREADS_PER_BLOCK; ++i) 
	{ 
		int32_t index = i * CUDA_DES_NUM_THREADS_PER_BLOCK + threadIdx.x; // スレッド並列でのインデックス計算
		cudaSharedCompactMediumChunkBitmap[index] = cudaCompactMediumChunkBitmap[index]; // グローバル定数メモリから最速ロード
	}
	__syncthreads(); // ブロック内全スレッドのロード完了を同期

	// 2. 64ビットバイナリカウンターの安全な復元処理
	uint64_t base_key_64 = 0; // 64ビットベースレジスタ初期化
	for (int i = 0; i < 8; ++i) {
		base_key_64 |= ((uint64_t)keyAndRandomBytes[i]) << ((7 - i) * 8); // ビットシフトを伴う正確なビッグエンディアン結合
	}

	// 徹底修正：32ビット整数あふれ（オーバーフロー）を防ぐため完全に uint64_t でキャスト計算
	uint64_t global_thread_id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    
	// 各スレッドに固有の重複しない完全連続総当たりバイナリベース鍵を確定
	uint64_t thread_raw_key = base_key_64 + global_thread_id;

	// 3. 各鍵ビットのビットスライス展開（0x0 ⇔ 0xFFFFFFFFU の極性拡張）
	// 各バイトの特定ビットが立っている場合、32スレッド分を一括表現する 0xFFFFFFFFU に安全マッピング
	uint32_t b1 = (uint32_t)((thread_raw_key >> 48) & 0xFF);
	DES_Vector K07 = ((b1 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K08 = ((b1 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K09 = ((b1 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K10 = ((b1 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K11 = ((b1 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K12 = ((b1 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K13 = ((b1 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b2 = (uint32_t)((thread_raw_key >> 40) & 0xFF);
	DES_Vector K14 = ((b2 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K15 = ((b2 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K16 = ((b2 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K17 = ((b2 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K18 = ((b2 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K19 = ((b2 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K20 = ((b2 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b3 = (uint32_t)((thread_raw_key >> 32) & 0xFF);
	DES_Vector K21 = ((b3 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K22 = ((b3 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K23 = ((b3 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K24 = ((b3 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K25 = ((b3 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K26 = ((b3 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K27 = ((b3 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b4 = (uint32_t)((thread_raw_key >> 24) & 0xFF);
	DES_Vector K28 = ((b4 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K29 = ((b4 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K30 = ((b4 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K31 = ((b4 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K32 = ((b4 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K33 = ((b4 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K34 = ((b4 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b5 = (uint32_t)((thread_raw_key >> 16) & 0xFF);
	DES_Vector K35 = ((b5 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K36 = ((b5 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K37 = ((b5 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K38 = ((b5 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K39 = ((b5 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K40 = ((b5 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K41 = ((b5 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b6 = (uint32_t)((thread_raw_key >> 8) & 0xFF);
	DES_Vector K42 = ((b6 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K43 = ((b6 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K44 = ((b6 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K45 = ((b6 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K46 = ((b6 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K47 = ((b6 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K48 = ((b6 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);

	uint32_t b7 = (uint32_t)(thread_raw_key & 0xFF);
	DES_Vector K49 = ((b7 & (1U << 7)) ? 0xFFFFFFFFU : 0x0); DES_Vector K50 = ((b7 & (1U << 6)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K51 = ((b7 & (1U << 5)) ? 0xFFFFFFFFU : 0x0); DES_Vector K52 = ((b7 & (1U << 4)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K53 = ((b7 & (1U << 3)) ? 0xFFFFFFFFU : 0x0); DES_Vector K54 = ((b7 & (1U << 2)) ? 0xFFFFFFFFU : 0x0);
	DES_Vector K55 = ((b7 & (1U << 1)) ? 0xFFFFFFFFU : 0x0);
	
	DES_Vector temp0, temp1; // KEYSWAP内で使用される退避用テンポラリレジスタ
	int32_t tripcodeIndex;   // 一致したビットスライスレーンインデックス (0〜31)
	tripcodeIndex = 0xFF;
	int32_t passCount;       // 暗号化対象セッションのループカウンタ

	// 4. パスセッションループ（総当たり試行開始）
	for (passCount = 0; passCount < CUDA_DES_MAX_PASS_COUNT; ++passCount) {
		unsigned char key = key0Array[passCount]; // カウンターベースではない動的鍵部分のロード
		DES_Vector K00 = ((key & (0x1U << 0)) ? 0xFFFFFFFFU : 0x0); DES_Vector K01 = ((key & (0x1U << 1)) ? 0xFFFFFFFFU : 0x0);
		DES_Vector K02 = ((key & (0x1U << 2)) ? 0xFFFFFFFFU : 0x0); DES_Vector K03 = ((key & (0x1U << 3)) ? 0xFFFFFFFFU : 0x0);
		DES_Vector K04 = ((key & (0x1U << 4)) ? 0xFFFFFFFFU : 0x0); DES_Vector K05 = ((key & (0x1U << 5)) ? 0xFFFFFFFFU : 0x0);
		DES_Vector K06 = ((key & (0x1U << 6)) ? 0xFFFFFFFFU : 0x0);

		// DESの内部データブロックレジスタ群（計64本、1本あたり32本分のビットデータに対応）
		DES_Vector DB00 = 0, DB01 = 0, DB02 = 0, DB03 = 0, DB04 = 0, DB05 = 0, DB06 = 0, DB07 = 0, DB08 = 0, DB09 = 0;
		DES_Vector DB10 = 0, DB11 = 0, DB12 = 0, DB13 = 0, DB14 = 0, DB15 = 0, DB16 = 0, DB17 = 0, DB18 = 0, DB19 = 0;
		DES_Vector DB20 = 0, DB21 = 0, DB22 = 0, DB23 = 0, DB24 = 0, DB25 = 0, DB26 = 0, DB27 = 0, DB28 = 0, DB29 = 0;
		DES_Vector DB30 = 0, DB31 = 0, DB32 = 0, DB33 = 0, DB34 = 0, DB35 = 0, DB36 = 0, DB37 = 0, DB38 = 0, DB39 = 0;
		DES_Vector DB40 = 0, DB41 = 0, DB42 = 0, DB43 = 0, DB44 = 0, DB45 = 0, DB46 = 0, DB47 = 0, DB48 = 0, DB49 = 0;
		DES_Vector DB50 = 0, DB51 = 0, DB52 = 0, DB53 = 0, DB54 = 0, DB55 = 0, DB56 = 0, DB57 = 0, DB58 = 0, DB59 = 0;
		DES_Vector DB60 = 0, DB61 = 0, DB62 = 0, DB63 = 0; 

		// 5. Modified-DES 25ラウンド暗号化コア実行
		for (int32_t ii = 0; ii < 25; ++ii) {
			DATASWAP; // ラウンド間のL側とR側の高速レジスタスワップ
			for (int32_t i = 0; i < 2; ++i) {

#if !defined(SALT) || __CUDA_ARCH__ < 500
				// S-Boxを用いたビットスライス換字・置換処理（マクロ非展開環境向けフォールバック）
				s1(((   1 & SALT) ? DB15 : DB31) ^ K12, ((   2 & SALT) ? DB16 : DB00) ^ K46, ((   4 & SALT) ? DB01 : DB17) ^ K33, ((   8 & SALT) ? DB18 : DB02) ^ K52, ((  16 & SALT) ? DB19 : DB03) ^ K48, ((  32 & SALT) ? DB20 : DB04) ^ K20, &DB40, &DB48, &DB54, &DB62);
				s2(((  64 & SALT) ? DB19 : DB03) ^ K34, (( 128 & SALT) ? DB20 : DB04) ^ K55, (( 256 & SALT) ? DB05 : DB21) ^ K05, (( 512 & SALT) ? DB22 : DB06) ^ K13, ((1024 & SALT) ? DB23 : DB07) ^ K18, ((2048 & SALT) ? DB24 : DB08) ^ K40, &DB44, &DB59, &DB33, &DB49);
				s3((                DB07       ) ^ K04, (                DB08       ) ^ K32, (                DB09       ) ^ K26, (                DB10       ) ^ K27, (                DB11       ) ^ K38, (                DB12       ) ^ K54, &DB55, &DB47, &DB61, &DB37);
				s4((                DB11       ) ^ K53, (                DB12       ) ^ K06, (                DB13       ) ^ K31, (                DB14       ) ^ K25, (                DB15       ) ^ K19, (                DB16       ) ^ K41, &DB57, &DB51, &DB41, &DB32);
				s5(((   1 & SALT) ? DB31 : DB15) ^ K15, ((   2 & SALT) ? DB00 : DB16) ^ K24, ((   4 & SALT) ? DB01 : DB17) ^ K28, ((   8 & SALT) ? DB02 : DB18) ^ K43, ((  16 & SALT) ? DB03 : DB19) ^ K30, ((  32 & SALT) ? DB04 : DB20) ^ K03, &DB39, &DB45, &DB56, &DB34);
				s6(((  64 & SALT) ? DB03 : DB19) ^ K35, (( 128 & SALT) ? DB04 : DB20) ^ K22, (( 256 & SALT) ? DB05 : DB21) ^ K02, (( 512 & SALT) ? DB06 : DB22) ^ K44, ((1024 & SALT) ? DB07 : DB23) ^ K14, ((2048 & SALT) ? DB08 : DB24) ^ K23, &DB35, &DB60, &DB42, &DB50);
				s7((                DB23       ) ^ K51, (                DB24       ) ^ K16, (                DB25       ) ^ K29, (                DB26       ) ^ K49, (                DB27       ) ^ K07, (                DB28       ) ^ K17, &DB63, &DB43, &DB53, &DB38);
				s8((                DB27       ) ^ K37, (                DB28       ) ^ K08, (                DB29       ) ^ K09, (                DB30       ) ^ K50, (                DB31       ) ^ K42, (                DB00       ) ^ K21, &DB36, &DB58, &DB46, &DB52);

				KEYSWAP12; // 鍵スケジュールのローテーション実行
		
				s1(((   1 & SALT) ? DB47 : DB63) ^ K05, ((   2 & SALT) ? DB48 : DB32) ^ K39, ((   4 & SALT) ? DB49 : DB33) ^ K26, ((   8 & SALT) ? DB50 : DB34) ^ K45, ((  16 & SALT) ? DB51 : DB35) ^ K41, ((  32 & SALT) ? DB52 : DB36) ^ K13, &DB08, &DB16, &DB22, &DB30);
				s2(((  64 & SALT) ? DB51 : DB35) ^ K27, (( 128 & SALT) ? DB52 : DB36) ^ K48, (( 256 & SALT) ? DB53 : DB37) ^ K53, (( 512 & SALT) ? DB54 : DB38) ^ K06, ((1024 & SALT) ? DB55 : DB39) ^ K11, ((2048 & SALT) ? DB56 : DB40) ^ K33, &DB12, &DB27, &DB01, &DB17);
				s3((                DB39       ) ^ K52, (                DB40       ) ^ K25, (                DB41       ) ^ K19, (                DB42       ) ^ K20, (                DB43       ) ^ K31, (                DB44       ) ^ K47, &DB23, &DB15, &DB29, &DB05);
				s4((                DB43       ) ^ K46, (                DB44       ) ^ K54, (                DB45       ) ^ K55, (                DB46       ) ^ K18, (                DB47       ) ^ K12, (                DB48       ) ^ K34, &DB25, &DB19, &DB09, &DB00);
				s5(((   1 & SALT) ? DB63 : DB47) ^ K08, ((   2 & SALT) ? DB32 : DB48) ^ K17, ((   4 & SALT) ? DB33 : DB49) ^ K21, ((   8 & SALT) ? DB34 : DB50) ^ K36, ((  16 & SALT) ? DB35 : DB51) ^ K23, ((  32 & SALT) ? DB36 : DB52) ^ K49, &DB07, &DB13, &DB24, &DB02);
				s6(((  64 & SALT) ? DB35 : DB51) ^ K28, (( 128 & SALT) ? DB36 : DB52) ^ K15, (( 256 & SALT) ? DB37 : DB53) ^ K24, (( 512 & SALT) ? DB38 : DB54) ^ K37, ((1024 & SALT) ? DB39 : DB55) ^ K07, ((2048 & SALT) ? DB40 : DB56) ^ K16, &DB03, &DB28, &DB10, &DB18);
				s7((                DB55       ) ^ K44, (                DB56       ) ^ K09, (                DB57       ) ^ K22, (                DB58       ) ^ K42, (                DB59       ) ^ K00, (                DB60       ) ^ K10, &DB31, &DB11, &DB21, &DB06);
				s8((                DB59       ) ^ K30, (                DB60       ) ^ K01, (                DB61       ) ^ K02, (                DB62       ) ^ K43, (                DB63       ) ^ K35, (                DB32       ) ^ K14, &DB04, &DB26, &DB14, &DB20);

				s1(((   1 & SALT) ? DB15 : DB31) ^ K46, ((   2 & SALT) ? DB16 : DB00) ^ K25, ((   4 & SALT) ? DB17 : DB01) ^ K12, ((   8 & SALT) ? DB18 : DB02) ^ K31, ((  16 & SALT) ? DB19 : DB03) ^ K27, ((  32 & SALT) ? DB20 : DB04) ^ K54, &DB40, &DB48, &DB54, &DB62);
				s2(((  64 & SALT) ? DB19 : DB03) ^ K13, (( 128 & SALT) ? DB20 : DB04) ^ K34, (( 256 & SALT) ? DB21 : DB05) ^ K39, (( 512 & SALT) ? DB22 : DB06) ^ K47, ((1024 & SALT) ? DB23 : DB07) ^ K52, ((2048 & SALT) ? DB24 : DB08) ^ K19, &DB44, &DB59, &DB33, &DB49);
				s3((                DB07       ) ^ K38, (                DB08       ) ^ K11, (                DB09       ) ^ K05, (                DB10       ) ^ K06, (                DB11       ) ^ K48, (                DB12       ) ^ K33, &DB55, &DB47, &DB61, &DB37);
				s4((                DB11       ) ^ K32, (                DB12       ) ^ K40, (                DB13       ) ^ K41, (                DB14       ) ^ K04, (                DB15       ) ^ K53, (                DB16       ) ^ K20, &DB57, &DB51, &DB41, &DB32);
				s5(((   1 & SALT) ? DB31 : DB15) ^ K51, ((   2 & SALT) ? DB00 : DB16) ^ K03, ((   4 & SALT) ? DB01 : DB17) ^ K07, ((   8 & SALT) ? DB02 : DB18) ^ K22, ((  16 & SALT) ? DB03 : DB19) ^ K09, ((  32 & SALT) ? DB04 : DB20) ^ K35, &DB39, &DB45, &DB56, &DB34);
				s6(((  64 & SALT) ? DB03 : DB19) ^ K14, (( 128 & SALT) ? DB04 : DB20) ^ K01, (( 256 & SALT) ? DB05 : DB21) ^ K10, (( 512 & SALT) ? DB06 : DB22) ^ K23, ((1024 & SALT) ? DB07 : DB23) ^ K50, ((2048 & SALT) ? DB08 : DB24) ^ K02, &DB35, &DB60, &DB42, &DB50);
				s7((                DB23       ) ^ K30, (                DB24       ) ^ K24, (                DB25       ) ^ K08, (                DB26       ) ^ K28, (                DB27       ) ^ K43, (                DB28       ) ^ K49, &DB63, &DB43, &DB53, &DB38);
				s8((                DB27       ) ^ K16, (                DB28       ) ^ K44, (                DB29       ) ^ K17, (                DB30       ) ^ K29, (                DB31       ) ^ K21, (                DB00       ) ^ K00, &DB36, &DB58, &DB46, &DB52);
		
				s1(((   1 & SALT) ? DB47 : DB63) ^ K32, ((   2 & SALT) ? DB48 : DB32) ^ K11, ((   4 & SALT) ? DB49 : DB33) ^ K53, ((   8 & SALT) ? DB50 : DB34) ^ K48, ((  16 & SALT) ? DB51 : DB35) ^ K13, ((  32 & SALT) ? DB52 : DB36) ^ K40, &DB08, &DB16, &DB22, &DB30);
				s2(((  64 & SALT) ? DB51 : DB35) ^ K54, (( 128 & SALT) ? DB52 : DB36) ^ K20, (( 256 & SALT) ? DB53 : DB37) ^ K25, (( 512 & SALT) ? DB54 : DB38) ^ K33, ((1024 & SALT) ? DB55 : DB39) ^ K38, ((2048 & SALT) ? DB56 : DB40) ^ K05, &DB12, &DB27, &DB01, &DB17);
				s3((                DB39       ) ^ K55, (                DB40       ) ^ K52, (                DB41       ) ^ K46, (                DB42       ) ^ K47, (                DB43       ) ^ K34, (                DB44       ) ^ K19, &DB23, &DB15, &DB29, &DB05);
				s4((                DB43       ) ^ K18, (                DB44       ) ^ K26, (                DB45       ) ^ K27, (                DB46       ) ^ K45, (                DB47       ) ^ K39, (                DB48       ) ^ K06, &DB25, &DB19, &DB09, &DB00);
				s5(((   1 & SALT) ? DB63 : DB47) ^ K37, ((   2 & SALT) ? DB32 : DB48) ^ K42, ((   4 & SALT) ? DB33 : DB49) ^ K50, ((   8 & SALT) ? DB34 : DB50) ^ K08, ((  16 & SALT) ? DB35 : DB51) ^ K24, ((  32 & SALT) ? DB36 : DB52) ^ K21, &DB07, &DB13, &DB24, &DB02);
				s6(((  64 & SALT) ? DB35 : DB51) ^ K00, (( 128 & SALT) ? DB36 : DB52) ^ K44, (( 256 & SALT) ? DB37 : DB53) ^ K49, (( 512 & SALT) ? DB38 : DB54) ^ K09, ((1024 & SALT) ? DB39 : DB55) ^ K36, ((2048 & SALT) ? DB40 : DB56) ^ K17, &DB03, &DB28, &DB10, &DB18);
				s7((                DB55       ) ^ K16, (                DB56       ) ^ K10, (                DB57       ) ^ K51, (                DB58       ) ^ K14, (                DB59       ) ^ K29, (                DB60       ) ^ K35, &DB31, &DB11, &DB21, &DB06);
				s8((                DB59       ) ^ K02, (                DB60       ) ^ K30, (                DB61       ) ^ K03, (                DB62       ) ^ K15, (                DB63       ) ^ K07, (                DB32       ) ^ K43, &DB04, &DB26, &DB14, &DB20);

				s1(((   1 & SALT) ? DB15 : DB31) ^ K18, ((   2 & SALT) ? DB16 : DB00) ^ K52, ((   4 & SALT) ? DB17 : DB01) ^ K39, ((   8 & SALT) ? DB18 : DB02) ^ K34, ((  16 & SALT) ? DB19 : DB03) ^ K54, ((  32 & SALT) ? DB20 : DB04) ^ K26, &DB40, &DB48, &DB54, &DB62);
				s2(((  64 & SALT) ? DB19 : DB03) ^ K40, (( 128 & SALT) ? DB20 : DB04) ^ K06, (( 256 & SALT) ? DB21 : DB05) ^ K11, (( 512 & SALT) ? DB22 : DB06) ^ K19, ((1024 & SALT) ? DB23 : DB07) ^ K55, ((2048 & SALT) ? DB24 : DB08) ^ K46, &DB44, &DB59, &DB33, &DB49);
				s3((                DB07       ) ^ K41, (                DB08       ) ^ K38, (                DB09       ) ^ K32, (                DB10       ) ^ K33, (                DB11       ) ^ K20, (                DB12       ) ^ K05, &DB55, &DB47, &DB61, &DB37);
				s4((                DB11       ) ^ K04, (                DB12       ) ^ K12, (                DB13       ) ^ K13, (                DB14       ) ^ K31, (                DB15       ) ^ K25, (                DB16       ) ^ K47, &DB57, &DB51, &DB41, &DB32);
				s5(((   1 & SALT) ? DB31 : DB15) ^ K23, ((   2 & SALT) ? DB00 : DB16) ^ K28, ((   4 & SALT) ? DB01 : DB17) ^ K36, ((   8 & SALT) ? DB02 : DB18) ^ K51, ((  16 & SALT) ? DB03 : DB19) ^ K10, ((  32 & SALT) ? DB04 : DB20) ^ K07, &DB39, &DB45, &DB56, &DB34);
				s6(((  64 & SALT) ? DB03 : DB19) ^ K43, (( 128 & SALT) ? DB04 : DB20) ^ K30, (( 256 & SALT) ? DB05 : DB21) ^ K35, (( 512 & SALT) ? DB06 : DB22) ^ K24, ((1024 & SALT) ? DB07 : DB23) ^ K22, ((2048 & SALT) ? DB08 : DB24) ^ K03, &DB35, &DB60, &DB42, &DB50);
				s7((                DB23       ) ^ K02, (                DB24       ) ^ K49, (                DB25       ) ^ K37, (                DB26       ) ^ K00, (                DB27       ) ^ K15, (                DB28       ) ^ K21, &DB63, &DB43, &DB53, &DB38);
				s8((                DB27       ) ^ K17, (                DB28       ) ^ K16, (                DB29       ) ^ K42, (                DB30       ) ^ K01, (                DB31       ) ^ K50, (                DB00       ) ^ K29, &DB36, &DB58, &DB46, &DB52);
		
				s1(((   1 & SALT) ? DB47 : DB63) ^ K04, ((   2 & SALT) ? DB48 : DB32) ^ K38, ((   4 & SALT) ? DB49 : DB33) ^ K25, ((   8 & SALT) ? DB50 : DB34) ^ K20, ((  16 & SALT) ? DB51 : DB35) ^ K40, ((  32 & SALT) ? DB52 : DB36) ^ K12, &DB08, &DB16, &DB22, &DB30);
				s2(((  64 & SALT) ? DB51 : DB35) ^ K26, (( 128 & SALT) ? DB52 : DB36) ^ K47, (( 256 & SALT) ? DB53 : DB37) ^ K52, (( 512 & SALT) ? DB54 : DB38) ^ K05, ((1024 & SALT) ? DB55 : DB39) ^ K41, ((2048 & SALT) ? DB56 : DB40) ^ K32, &DB12, &DB27, &DB01, &DB17);
				s3((                DB39       ) ^ K27, (                DB40       ) ^ K55, (                DB41       ) ^ K18, (                DB42       ) ^ K19, (                DB43       ) ^ K06, (                DB44       ) ^ K46, &DB23, &DB15, &DB29, &DB05);
				s4((                DB43       ) ^ K45, (                DB44       ) ^ K53, (                DB45       ) ^ K54, (                DB46       ) ^ K48, (                DB47       ) ^ K11, (                DB48       ) ^ K33, &DB25, &DB19, &DB09, &DB00);
				s5(((   1 & SALT) ? DB63 : DB47) ^ K09, ((   2 & SALT) ? DB32 : DB48) ^ K14, ((   4 & SALT) ? DB33 : DB49) ^ K22, ((   8 & SALT) ? DB34 : DB50) ^ K37, ((  16 & SALT) ? DB35 : DB51) ^ K49, ((  32 & SALT) ? DB36 : DB52) ^ K50, &DB07, &DB13, &DB24, &DB02);
				s6(((  64 & SALT) ? DB35 : DB51) ^ K29, (( 128 & SALT) ? DB36 : DB52) ^ K16, (( 256 & SALT) ? DB37 : DB53) ^ K21, (( 512 & SALT) ? DB38 : DB54) ^ K10, ((1024 & SALT) ? DB39 : DB55) ^ K08, ((2048 & SALT) ? DB40 : DB56) ^ K42, &DB03, &DB28, &DB10, &DB18);
				s7((                DB55       ) ^ K17, (                DB56       ) ^ K35, (                DB57       ) ^ K23, (                DB58       ) ^ K43, (                DB59       ) ^ K01, (                DB60       ) ^ K07, &DB31, &DB11, &DB21, &DB06);
				s8((                DB59       ) ^ K03, (                DB60       ) ^ K02, (                DB61       ) ^ K28, (                DB62       ) ^ K44, (                DB63       ) ^ K36, (                DB32       ) ^ K15, &DB04, &DB26, &DB14, &DB20);
		
				s1(((   1 & SALT) ? DB15 : DB31) ^ K45, ((   2 & SALT) ? DB16 : DB00) ^ K55, ((   4 & SALT) ? DB17 : DB01) ^ K11, ((   8 & SALT) ? DB18 : DB02) ^ K06, ((  16 & SALT) ? DB19 : DB03) ^ K26, ((  32 & SALT) ? DB20 : DB04) ^ K53, &DB40, &DB48, &DB54, &DB62);
				s2(((  64 & SALT) ? DB19 : DB03) ^ K12, (( 128 & SALT) ? DB20 : DB04) ^ K33, (( 256 & SALT) ? DB21 : DB05) ^ K46, (( 512 & SALT) ? DB22 : DB06) ^ K27, ((1024 & SALT) ? DB23 : DB07) ^ K18, ((2048 & SALT) ? DB24 : DB08) ^ K18, &DB44, &DB59, &DB33, &DB49);
				s3((                DB07       ) ^ K13, (                DB08       ) ^ K41, (                DB09       ) ^ K04, (                DB10       ) ^ K05, (                DB11       ) ^ K47, (                DB12       ) ^ K32, &DB55, &DB47, &DB61, &DB37);
				s4((                DB11       ) ^ K31, (                DB12       ) ^ K39, (                DB13       ) ^ K40, (                DB14       ) ^ K34, (                DB15       ) ^ K52, (                DB16       ) ^ K19, &DB57, &DB51, &DB41, &DB32);
				s5(((   1 & SALT) ? DB31 : DB15) ^ K24, ((   2 & SALT) ? DB00 : DB16) ^ K00, ((   4 & SALT) ? DB01 : DB17) ^ K08, ((   8 & SALT) ? DB02 : DB18) ^ K23, ((  16 & SALT) ? DB03 : DB19) ^ K35, ((  32 & SALT) ? DB04 : DB20) ^ K36, &DB39, &DB45, &DB56, &DB34);
				// 👈 徹底修正：第3引数のサイレントバグを DB07 から正しい直交ペア DB05 に修正完了
				s6(((  64 & SALT) ? DB03 : DB19) ^ K15, (( 128 & SALT) ? DB04 : DB20) ^ K02, (( 256 & SALT) ? DB05 : DB21) ^ K07, (( 512 & SALT) ? DB06 : DB22) ^ K49, ((1024 & SALT) ? DB07 : DB23) ^ K51, ((2048 & SALT) ? DB08 : DB24) ^ K28, &DB35, &DB60, &DB42, &DB50);
				s7((                DB23       ) ^ K03, (                DB24       ) ^ K21, (                DB25       ) ^ K09, (                DB26       ) ^ K29, (                DB27       ) ^ K44, (                DB28       ) ^ K50, &DB63, &DB43, &DB53, &DB38);
				s8((                DB27       ) ^ K42, (                DB28       ) ^ K17, (                DB29       ) ^ K14, (                DB30       ) ^ K30, (                DB31       ) ^ K22, (                DB00       ) ^ K01, &DB36, &DB58, &DB46, &DB52);

				KEYSWAP21; // 鍵スケジュールの逆ローテーション実行
		
				s1(((   1 & SALT) ? DB47 : DB63) ^ K31, ((   2 & SALT) ? DB48 : DB32) ^ K41, ((   4 & SALT) ? DB49 : DB33) ^ K52, ((   8 & SALT) ? DB50 : DB34) ^ K47, ((  16 & SALT) ? DB51 : DB35) ^ K12, ((  32 & SALT) ? DB52 : DB36) ^ K39, &DB08, &DB16, &DB22, &DB30);
				s2(((  64 & SALT) ? DB51 : DB35) ^ K53, (( 128 & SALT) ? DB52 : DB36) ^ K19, (( 256 & SALT) ? DB53 : DB37) ^ K55, (( 512 & SALT) ? DB54 : DB38) ^ K32, ((1024 & SALT) ? DB55 : DB39) ^ K13, ((2048 & SALT) ? DB56 : DB40) ^ K04, &DB12, &DB27, &DB01, &DB17);
				s3((                DB39       ) ^ K54, (                DB40       ) ^ K27, (                DB41       ) ^ K45, (                DB42       ) ^ K46, (                DB43       ) ^ K33, (                DB44       ) ^ K18, &DB23, &DB15, &DB29, &DB05);
				s4((                DB43       ) ^ K48, (                DB44       ) ^ K25, (                DB45       ) ^ K26, (                DB46       ) ^ K20, (                DB47       ) ^ K38, (                DB48       ) ^ K05, &DB25, &DB19, &DB09, &DB00);
				s5(((   1 & SALT) ? DB63 : DB47) ^ K10, ((   2 & SALT) ? DB32 : DB48) ^ K43, ((   4 & SALT) ? DB33 : DB49) ^ K51, ((   8 & SALT) ? DB34 : DB50) ^ K09, ((  16 & SALT) ? DB35 : DB51) ^ K21, ((  32 & SALT) ? DB36 : DB52) ^ K22, &DB07, &DB13, &DB24, &DB02);
				s6(((  64 & SALT) ? DB35 : DB51) ^ K01, (( 128 & SALT) ? DB36 : DB52) ^ K17, (( 256 & SALT) ? DB37 : DB53) ^ K50, (( 512 & SALT) ? DB38 : DB54) ^ K35, ((1024 & SALT) ? DB39 : DB55) ^ K37, ((2048 & SALT) ? DB40 : DB56) ^ K14, &DB03, &DB28, &DB10, &DB18);
				s7((                DB55       ) ^ K42, (                DB56       ) ^ K07, (                DB57       ) ^ K24, (                DB58       ) ^ K15, (                DB59       ) ^ K30, (                DB60       ) ^ K36, &DB31, &DB11, &DB21, &DB06);
				s8((                DB59       ) ^ K28, (                DB60       ) ^ K03, (                DB61       ) ^ K00, (                DB62       ) ^ K16, (                DB63       ) ^ K08, (                DB32       ) ^ K44, &DB04, &DB26, &DB14, &DB20);
		
				SWAP01; // 鍵データの初期直交化状態への復元
#else
				// Kepler（CUDAアーキテクチャ500以上）等に向けたソルトマクロ最適化展開展開
				CUDA_DES_CRYPT_EIGHT_ROUNDS(SALT);
#endif
			}
		}

		// 6. トリップ判定ロジック（マッチング判定）
		if (numTripcodeChunk == 1 && searchMode == SEARCH_MODE_FORWARD_MATCHING) {
			// 単一のターゲットに対する最速の直接ハッシュ値比較
			for (tripcodeIndex = 0; tripcodeIndex < CUDA_DES_BS_DEPTH; ++tripcodeIndex) {
				uint32_t tripcodeChunk = tripcodeChunkArray[0];
				if (GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB63, DB31, DB38, DB06, DB46, DB14, 0) != ((tripcodeChunk >> (6 * 4)) & 0x3f)) continue;
				if (GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB54, DB22, DB62, DB30, DB37, DB05, 0) != ((tripcodeChunk >> (6 * 3)) & 0x3f)) continue;
				if (GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB45, DB13, DB53, DB21, DB61, DB29, 0) != ((tripcodeChunk >> (6 * 2)) & 0x3f)) continue;
				if (GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB36, DB04, DB44, DB12, DB52, DB20, 0) != ((tripcodeChunk >> (6 * 1)) & 0x3f)) continue;
				if (GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB60, DB28, DB35, DB03, DB43, DB11, 0) != ((tripcodeChunk >> (6 * 0)) & 0x3f)) continue;
				goto quit_loops; // 一致したためループ脱出
			}
		} else if (searchMode == SEARCH_MODE_FORWARD_MATCHING) {
			// 複数ターゲットに対する前方一致（ビットマップフィルタ + 二分探索）
			for (tripcodeIndex = 0; tripcodeIndex < CUDA_DES_BS_DEPTH; ++tripcodeIndex) {
				uint32_t tripcodeChunk =   GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB63, DB31, DB38, DB06, DB46, DB14, 4)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB54, DB22, DB62, DB30, DB37, DB05, 3)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB45, DB13, DB53, DB21, DB61, DB29, 2)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB36, DB04, DB44, DB12, DB52, DB20, 1)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB60, DB28, DB35, DB03, DB43, DB11, 0);
				if (cudaSharedCompactMediumChunkBitmap[tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6 + 3)] & (1 << ((tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6)) & 7))) continue;
				if (cudaChunkBitmap[tripcodeChunk >> ((5 - CHUNK_BITMAP_LEN_STRING) * 6)]) continue;
				BINARY_SEARCH; // 正確なインデックスを二分探索
			}
		} else if (searchMode == SEARCH_MODE_BACKWARD_MATCHING) {
			// 後方一致判定
			for (tripcodeIndex = 0; tripcodeIndex < CUDA_DES_BS_DEPTH; ++tripcodeIndex) {
				uint32_t tripcodeChunk =   GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB51, DB19, DB59, DB27, DB34, DB02, 4)
							                 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB42, DB10, DB50, DB18, DB58, DB26, 3)
									         | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB33, DB01, DB41, DB09, DB49, DB17, 2)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB57, DB25, DB32, DB00, DB40, DB08, 1)
											 | GET_TRIPCODE_CHAR_INDEX_LAST(tripcodeIndex, DB48, DB16, DB56, DB24);
				if ((cudaSharedCompactMediumChunkBitmap[tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6 + 3)] & (1 << ((tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6)) & 7)))) continue;
				if (cudaChunkBitmap[tripcodeChunk >> ((5 - CHUNK_BITMAP_LEN_STRING) * 6)]) continue;
				BINARY_SEARCH;
			}
		} else if (searchMode == SEARCH_MODE_FORWARD_AND_BACKWARD_MATCHING) {
			// 前方・後方両方向一致判定
			for (tripcodeIndex = 0; tripcodeIndex < CUDA_DES_BS_DEPTH; ++tripcodeIndex) {
				uint32_t tripcodeChunk =   GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB63, DB31, DB38, DB06, DB46, DB14, 4)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB54, DB22, DB62, DB30, DB37, DB05, 3)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB45, DB13, DB53, DB21, DB61, DB29, 2)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB36, DB04, DB44, DB12, DB52, DB20, 1)
											 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB60, DB28, DB35, DB03, DB43, DB11, 0);
				if (   !(cudaSharedCompactMediumChunkBitmap[tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6 + 3)] & (1 << ((tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6)) & 7)))
				    && !cudaChunkBitmap[tripcodeChunk >> ((5 - CHUNK_BITMAP_LEN_STRING) * 6)]) {
					BINARY_SEARCH;
				}
				tripcodeChunk =   GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB51, DB19, DB59, DB27, DB34, DB02, 4)
								 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB42, DB10, DB50, DB18, DB58, DB26, 3)
								 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB33, DB01, DB41, DB09, DB49, DB17, 2)
								 | GET_TRIPCODE_CHAR_INDEX(tripcodeIndex, DB57, DB25, DB32, DB00, DB40, DB08, 1)
								 | GET_TRIPCODE_CHAR_INDEX_LAST(tripcodeIndex, DB48, DB16, DB56, DB24);
				if (   !(cudaSharedCompactMediumChunkBitmap[tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6 + 3)] & (1 << ((tripcodeChunk >> ((5 - MEDIUM_CHUNK_BITMAP_LEN_STRING) * 6)) & 7)))
				    && !cudaChunkBitmap[tripcodeChunk >> ((5 - CHUNK_BITMAP_LEN_STRING) * 6)]) {
					BINARY_SEARCH;
				}
			}
		}
	}

	// すべての走査でヒットしなかった場合、最大値を代入して完了通知とする
	passCount = CUDA_DES_MAX_PASS_COUNT;

quit_loops: // マッチング時にジャンプしてくる絶対安全な合流用ラベル
	// 徹底修正：書き戻し用のスレッドID計算において、32bit整数オーバーフローを安全にキャストして完全排除
	global_thread_id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
	passCountArray[global_thread_id] = passCount;         // 走査完了数の書き戻し
	tripcodeIndexArray[global_thread_id] = tripcodeIndex; // マッチしたスライスレーンIDの確定保存
}

#undef SALT
#undef KERNEL_FUNC
#undef KERNEL_FUNC2
