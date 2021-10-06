//
//  CABAC.hpp
//  test
//
//  Created by Alex on 2021/10/4.
//

#ifndef CABAC_hpp
#define CABAC_hpp

#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
#include <assert.h>
#include <sstream>

#include "CABAC_ArithmeticEncoder.h"
#include "CABAC_ArithmeticDecoder.h"
#include "CABAC_BitstreamFile.h"
#include "ContextModel.h"
#include "CommonDef.h"

using namespace std;

inline int str2int(const string& s){
    istringstream is(s);
    int res;
    is>>res;
    return res;
}

const char* str2pchar(const string& s){
    const char *p = s.c_str();
    return p;
}

void ContextUpdate(long int& context, int newC, const int n_context_model){
    context = (context << 1) + newC;
    while (context >= n_context_model){
        context -= n_context_model;
    }
    if (n_context_model == 1){
        context = 0;
    }
}

unsigned long int encode(const int n_context, const string file, const string outfile){
    CABAC_BitstreamFile outStream;
    if (!outStream.openOutputFile(str2pchar(outfile))){
        fprintf(stderr, "\nfailed to open bitstream file\n");
        return -1;
    }
    int n_context_model = 2 << n_context;
    ContextModel* ctx = (ContextModel*)malloc(sizeof(ContextModel) * (n_context_model));
    long int context = 0;
    for (int i=0; i < n_context_model; ++i){
        ctx[i] = ContextModel();
    }
    CABAC_ArithmeticEncoder AE(&outStream);
    AE.start();
    FILE *fp;
    fp = fopen(str2pchar(file), "rb");
    char ch;
    unsigned long int ct = 0;
    while (1){
        ch = fgetc(fp);
        if (ch == EOF){
            break;
        }
        ct += 1;
        int val = ch -'0';
        AE.encodeBin(val,  &ctx[context]);
        ContextUpdate(context, val, n_context_model);
    }
    AE.finish();
    outStream.closeFile();
    fclose(fp);
#if RWTH_TRACE_CABAC_STATES && RWTH_TRACE_CABAC_TO_FILE
    FILE *m_cTraceCabacStatFile = fopen("CabacStats.log", "w");
    for (int i=0; i < n_context_model; ++i){
        ctx[i].traceStatesToFile(m_cTraceCabacStatFile);
    }
    fclose(m_cTraceCabacStatFile);
#endif
    return ct;
}

void decode(const unsigned long int ct, const int n_context, const string file, const string outfile){
    CABAC_BitstreamFile inStream;
    if (!inStream.openInputFile(str2pchar(outfile))){
      fprintf(stderr, "\nfailed to open bitstream file\n");
      return;
    }
    int n_context_model = 2 << n_context;
    ContextModel* ctx = (ContextModel*)malloc(sizeof(ContextModel) * (n_context_model));
    long int context = 0;
    for (int i=0; i < n_context_model; ++i){
        ctx[i] = ContextModel();
    }
    FILE *fp;
    fp = fopen(str2pchar(file), "wb");
    CABAC_ArithmeticDecoder AD(&inStream);
    AD.start();
    unsigned int uiBit;
    for (int i=0; i < ct; ++i){
        AD.decodeBin(uiBit, &ctx[context]);
        fputc(uiBit+'0', fp);
        ContextUpdate(context, (int)uiBit, n_context_model);
    }
    AD.finish();
    inStream.closeFile();
    fclose(fp);
}

#endif /* CABAC_hpp */
