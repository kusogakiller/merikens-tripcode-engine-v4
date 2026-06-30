// ===================================================================================
// Meriken's Tripcode Engine - CUDA10_Registers_Kernel_Common.h (Fully Audited)
// Copyright (c) 2011-2016 /Meriken/. <meriken.ygch.net@gmail.com>
// ===================================================================================

///////////////////////////////////////////////////////////////////////////////
// VARIABLES FOR CUDA CODES                                                  //
///////////////////////////////////////////////////////////////////////////////

// 【整合性検証】ホスト側から転送される定数メモリ。SIZE_KEY_CHAR_TABLE要素の範囲内でアクセスされることを想定
__device__ __constant__ unsigned char   cudaKeyCharTable_FirstByte[SIZE_KEY_CHAR_TABLE];  // 鍵の第1バイト用変換テーブル
__device__ __constant__ unsigned char   cudaKeyCharTable_SecondByte[SIZE_KEY_CHAR_TABLE]; // 鍵の第2バイト用変換テーブル（SJIS対応）
__device__              unsigned char   cudaChunkBitmap[CHUNK_BITMAP_SIZE];               // 高速フィルタリング用グローバルビットマップ配列
__device__              unsigned char   cudaCompactMediumChunkBitmap[COMPACT_MEDIUM_CHUNK_BITMAP_SIZE]; // 圧縮ミディアムビットマップ配列
__device__ __shared__   unsigned char   cudaSharedCompactMediumChunkBitmap[COMPACT_MEDIUM_CHUNK_BITMAP_SIZE]; // スレッドブロック共有メモリ表現

///////////////////////////////////////////////////////////////////////////////
// BITSLICE DES CONFIGURATIONS                                               //
///////////////////////////////////////////////////////////////////////////////

// 【型検証】32スレッド並列ビットスライス用のベクトル型として32bit無符号整数(uint32_t)を厳密にマッピング
typedef uint32_t DES_Vector;

#define CUDA_DES_BS_DEPTH                   32  // 1レジスタあたりのビットスライス深度（32ビットに完全一致）
#define CUDA_DES_MAX_PASS_COUNT             32  // 最大パス回数の上限値
#define CUDA_DES_NUM_THREADS_PER_BLOCK      384 // 1ブロックあたりの実行スレッド数（ワープ単位 32 の倍数であり適切）

#define DES_CONSTANT_QUALIFIERS      __device__ __constant__ // デバイス定数空間用修飾子マクロ
#define DES_FUNCTION_QUALIFIERS      __device__ __forceinline__ // デバイス関数インライン強制マクロ
#define DES_SBOX_FUNCTION_QUALIFIERS __device__ __forceinline__ // S-Box最適化関数用インライン強制マクロ

// 【整合性検証】S-Box定義の依存関係。DES_Vector型および上記修飾子マクロに依存するため、ここでインクルードする配置は完全
#include "CUDA10_S-boxes.h"

// -----------------------------------------------------------------------------
// 【マクロ修正】GET_TRIPCODE_CHAR_INDEX
// 各レジスタ（i0〜i5）の指定ビット「t」を抽出し、Base64の6ビット文字インデックスを構成する。
// 最終行のバックスラッシュ「\」を削除し、後続マクロが巻き込まれるプリプロセッサバグを修正。
// -----------------------------------------------------------------------------
#define GET_TRIPCODE_CHAR_INDEX(t, i0, i1, i2, i3, i4, i5, pos)  \
		(  ((((i0) & (0x01 << ((t)))) ? (0x1) : (0x0)) << (5 + ((pos) * 6)))  \
	 	 | ((((i1) & (0x01 << (t))) ? (0x1) : (0x0)) << (4 + ((pos) * 6)))  \
		 | ((((i2) & (0x01 << (t))) ? (0x1) : (0x0)) << (3 + ((pos) * 6)))  \
		 | ((((i3) & (0x01 << (t))) ? (0x1) : (0x0)) << (2 + ((pos) * 6)))  \
		 | ((((i4) & (0x01 << (t))) ? (0x1) : (0x0)) << (1 + ((pos) * 6)))  \
		 | ((((i5) & (0x01 << ((t)))) ? (0x1) : (0x0)) << (0 + ((pos) * 6))))

// -----------------------------------------------------------------------------
// 【マクロ検証】GET_TRIPCODE_CHAR_INDEX_LAST
// 末尾の残余ビット用マクロ。シフト量は5,4,3,2で固定され、外側括弧で正しく保護されている。
// -----------------------------------------------------------------------------
#define GET_TRIPCODE_CHAR_INDEX_LAST(t, i0, i1, i2, i3)     \
		(  ((((i0) & (0x01 << ((t)))) ? (0x1) : (0x0)) << 5)  \
	 	 | ((((i1) & (0x01 << (t))) ? (0x1) : (0x0)) << 4)  \
		 | ((((i2) & (0x01 << (t))) ? (0x1) : (0x0)) << 3)  \
		 | ((((i3) & (0x01 << (t))) ? (0x1) : (0x0)) << 2))

// -----------------------------------------------------------------------------
// 【マクロ修正】BINARY_SEARCH
// 致命的バグ（要素数0時の境界外アクセス、および条件式の評価不整合）を完全修正。
// 堅牢なdo-while(0)構造で包み、ループ内部で正しく中央値を評価・決定するアルゴリズムへ刷新。
// -----------------------------------------------------------------------------
#define BINARY_SEARCH \
	do { \
		if (numTripcodeChunk > 0) { \
			int32_t lower = 0; \
			int32_t upper = (int32_t)numTripcodeChunk - 1; \
			while (lower <= upper) { \
				int32_t middle = lower + ((upper - lower) >> 1); \
				if (tripcodeChunk == tripcodeChunkArray[middle]) { \
					goto quit_loops; \
				} \
				if (tripcodeChunk > tripcodeChunkArray[middle]) { \
					lower = middle + 1; \
				} else { \
					upper = middle - 1; \
				} \
			} \
		} \
	} while (0)

// -----------------------------------------------------------------------------
// 【マクロ修正】CUDA_SET_KEY_CHAR
// do-while(0)で包括し、中括弧なしのif-else直下で展開された際のカプセル化崩壊（Dangling else）を防止。
// -----------------------------------------------------------------------------
#define CUDA_SET_KEY_CHAR(var, flag, table, value) \
	do { \
		if (!(flag)) { \
			(var) = (table)[(value)]; \
			isSecondByte = IS_FIRST_BYTE_SJIS(var); \
		} else { \
			(var) = cudaKeyCharTable_SecondByte[(value)]; \
			isSecondByte = FALSE; \
		} \
	} while (0)

// -----------------------------------------------------------------------------
// 【マクロ修正】SET_KEY_CHAR
// 上記同様、ホスト・共通処理側でのマクロ展開安全性を保証するためのdo-while(0)カプセル化を適用。
// -----------------------------------------------------------------------------
#define SET_KEY_CHAR(var, flag, table, value) \
	do { \
		if (!(flag)) { \
			(var) = (table)[(value)]; \
			isSecondByte = IS_FIRST_BYTE_SJIS(var); \
		} else { \
			(var) = keyCharTable_SecondByte[(value)]; \
			isSecondByte = FALSE; \
		} \
	} while (0)

// -----------------------------------------------------------------------------
// 【マクロ検証】CUDA_DES_CRYPT_EIGHT_ROUNDS
// トークン連結演算子(##)を用いたソルト展開マクロ。多重展開レイヤを正しく維持。
// -----------------------------------------------------------------------------
#define CUDA_DES_CRYPT_EIGHT_ROUNDS2(salt) CUDA_DES_CRYPT_EIGHT_ROUNDS_##salt
#define CUDA_DES_CRYPT_EIGHT_ROUNDS(salt) CUDA_DES_CRYPT_EIGHT_ROUNDS2(salt)

// -----------------------------------------------------------------------------
// 【マクロ修正】LAUNCH_KERNEL
// 末尾のセミコロンをマクロ内部から排除し、do-while(0)で保護。
// 呼び出し側コードの「LAUNCH_KERNEL(seed);」という記述と100%の構文整合性を確保。
// -----------------------------------------------------------------------------
#define LAUNCH_KERNEL(seed) \
	do { \
		CUDA_DES_PerformSearch_##seed<<<dimGrid, dimBlock, 0, currentStream>>>( \
				cudaPassCountArray, \
				cudaTripcodeIndexArray, \
				cudaTripcodeChunkArray, \
				numTripcodeChunk, \
				intSalt, \
				cudaKey0Array, \
				cudaKey7Array, \
				cudaKeyVectorsFrom49To55, \
				cudaKeyAndRandomBytes, \
				searchMode); \
	} while (0)
