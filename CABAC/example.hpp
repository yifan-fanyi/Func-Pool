//
//  example.hpp
//  test
//
//  Created by Alex on 2021/10/4.
//

#ifndef example_hpp
#define example_hpp

#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
#include <assert.h>

#include "CABAC_ArithmeticEncoder.h"
#include "CABAC_ArithmeticDecoder.h"
#include "CABAC_BitstreamFile.h"
#include "ContextModel.h"
#include "CommonDef.h"
using namespace std;

void codeToFile()
{
  // Create and open the bitstream to write to
  CABAC_BitstreamFile outStream;
  if (!outStream.openOutputFile("str.bin"))
  {
    fprintf(stderr, "\nfailed to open bitstream file `str.bin' for writing\n");
    return;
  }
  printf("Opened 'str.bin' for writing.\n");
  // Create the arithmetic coder (and provide the bitstream it shall write to)
  CABAC_ArithmeticEncoder arithmeticEncoder(&outStream);
  arithmeticEncoder.start();
  
  // Create a context, initialize it and code 001011 to is
  ContextModel ctx0;
  ctx0.init(0,20);    // Optional: Initialize the context to something other than equal probability.
  arithmeticEncoder.encodeBin( 0, &ctx0 );
  arithmeticEncoder.encodeBin( 0, &ctx0 );
  arithmeticEncoder.encodeBin( 1, &ctx0 );
  arithmeticEncoder.encodeBin( 0, &ctx0 );
  arithmeticEncoder.encodeBin( 1, &ctx0 );
  arithmeticEncoder.encodeBin( 1, &ctx0 );
    
  // Encode 5 EP bits (10010)
  arithmeticEncoder.encodeBinEP(1);
  arithmeticEncoder.encodeBinEP(0);
  arithmeticEncoder.encodeBinEP(0);
  arithmeticEncoder.encodeBinEP(1);
  arithmeticEncoder.encodeBinEP(0);

  // Encode the same 5 bits (10010) using the encodeBinsEP function
  arithmeticEncoder.encodeBinsEP(18, 5);

  // Create another context and code 110111 to it
  ContextModel ctx1;
  arithmeticEncoder.encodeBin( 1, &ctx1 );
  arithmeticEncoder.encodeBin( 1, &ctx1 );
  arithmeticEncoder.encodeBin( 0, &ctx1 );
  arithmeticEncoder.encodeBin( 1, &ctx1 );
  arithmeticEncoder.encodeBin( 1, &ctx1 );
  arithmeticEncoder.encodeBin( 1, &ctx1 );

#if RWTH_CABAC_FIXED_PROBABILITY
  // Encode some bits with a fixed probabilty
  arithmeticEncoder.encodeBinProb(1, 10);
  arithmeticEncoder.encodeBinProb(0, 10);
  arithmeticEncoder.encodeBinProb(0, 10);

  arithmeticEncoder.encodeBinProb(1, 30);
  arithmeticEncoder.encodeBinProb(1, 30);
  arithmeticEncoder.encodeBinProb(0, 30);
#endif

  // Finish coding
  arithmeticEncoder.finish();
  outStream.closeFile();
    
#if RWTH_TRACE_CABAC_STATES && RWTH_TRACE_CABAC_TO_FILE
  // Coding is complete. Write CABAC stats to file
  // Open the output file to write the CABAC state counters to
  FILE *m_cTraceCabacStatFile = fopen("CabacStats.log", "w");
  // Trace all ctx to the file
  ctx0.traceStatesToFile(m_cTraceCabacStatFile);
  ctx1.traceStatesToFile(m_cTraceCabacStatFile);
  // Close the CABAC stat file
  fclose(m_cTraceCabacStatFile);
#endif
}

void decodeFromFile()
{
  // Create and open the input bitstream
  CABAC_BitstreamFile inStream;
  if (!inStream.openInputFile("str.bin"))
  {
    fprintf(stderr, "\nfailed to open bitstream file `str.bin' for reading\n");
    return;
  }
  printf("Opened 'str.bin' for reading.\n");
  
  CABAC_ArithmeticDecoder arithmeticDecoder(&inStream);
  arithmeticDecoder.start();
  unsigned int uiBit;

  // Create a context, initialize it, and decode 6 bits from it. (Should be 001011)
  ContextModel ctx0;
  ctx0.init(0,20);    // Optional: Initialize the context to something other than equal probability.
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 0);
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 0);
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 0);
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx0); assert(uiBit == 1);

  // Decode 4 EP bits (should be 10010)
  arithmeticDecoder.decodeBinEP(uiBit); assert(uiBit == 1);
  arithmeticDecoder.decodeBinEP(uiBit); assert(uiBit == 0);
  arithmeticDecoder.decodeBinEP(uiBit); assert(uiBit == 0);
  arithmeticDecoder.decodeBinEP(uiBit); assert(uiBit == 1);
  arithmeticDecoder.decodeBinEP(uiBit); assert(uiBit == 0);

  // Decode the same 5 bits (10010) using the decodeBinsEP function
  unsigned int uiVal;
  arithmeticDecoder.decodeBinsEP(uiVal, 5);
  assert(uiVal==18);

  // Create another context and decode six bits (should be 110111)
  ContextModel ctx1;
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 0);
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 1);
  arithmeticDecoder.decodeBin(uiBit, &ctx1); assert(uiBit == 1);
    
#if RWTH_CABAC_FIXED_PROBABILITY
  // Encode some bits with a fixed probabilty
  arithmeticDecoder.decodeBinProb(uiBit, 10); assert(uiBit==1);
  arithmeticDecoder.decodeBinProb(uiBit, 10); assert(uiBit==0);
  arithmeticDecoder.decodeBinProb(uiBit, 10); assert(uiBit==0);

  arithmeticDecoder.decodeBinProb(uiBit, 30); assert(uiBit==1);
  arithmeticDecoder.decodeBinProb(uiBit, 30); assert(uiBit==1);
  arithmeticDecoder.decodeBinProb(uiBit, 30); assert(uiBit==0);
#endif

  arithmeticDecoder.finish();
  inStream.closeFile();
}

#endif /* example_hpp */
