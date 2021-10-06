/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#pragma once

#include "CABAC_BitstreamFile.h"
#include "ContextModel.h"
#include "CommonDef.h"
#include "assert.h"
#if RWTH_TRACE_CABAC_TO_FILE
#include <cstdio>
#endif

/** The arithmetic coder engine class
  *
  * This class performes the arithmetic coding and writes the resulting bits into a bitstream.
  */

class CABAC_ArithmeticEncoder
{
public:
  CABAC_ArithmeticEncoder( CABAC_BitstreamFile* ptCabacBitstream=NULL );
  ~CABAC_ArithmeticEncoder();
  void setBitstream(CABAC_BitstreamFile* ptCabacBitstream);
  CABAC_BitstreamFile* getBitstream() { assert(m_ptBitstream); return m_ptBitstream; };

  void  start            ();
  void  finish           ();

  void  encodeBin         ( unsigned int  binValue,  ContextModel *rcCtxModel );
  void  encodeBinEP       ( unsigned int  binValue                            );
  void  encodeBinsEP      ( unsigned int  binValues, int numBins              );
  
#if RWTH_CABAC_FIXED_PROBABILITY
  // Encode a bit with a certain ficed probability. No context deeded/updated
  void  encodeBinProb     ( unsigned int  binValue, unsigned int uiProbability        );
#endif

  unsigned int  getBinsCoded () { return m_uiBinsCoded; }

protected:
  void  encodeBinTrm     ( unsigned int  binValue                            );

  CABAC_BitstreamFile *m_ptBitstream;

  unsigned int        m_uiLow;
  unsigned int        m_uiRange;
  unsigned int        m_bufferedByte;
  int                 m_numBufferedBytes;
  int                 m_bitsLeft;
  unsigned int        m_uiBinsCoded;

  void testAndWriteOut();
  void writeOut();

  const static unsigned char  sm_aucLPSTable[64][4];
  const static unsigned char  sm_aucRenormTable[32];
#if RWTH_CABAC_FIXED_PROBABILITY
  const static unsigned char  sm_aucLPSTProbTable[49][4];
#endif
#if RWTH_TRACE_CABAC_TO_FILE
  FILE * c;
#endif
};

