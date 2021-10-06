/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.  
 *
 * Copyright (c) 2016-2017, Institut für Nachrichtentechnik, RWTH Aachen University
 *
 * Christian Feldmann
 *
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
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
 
#include "CABAC.hpp"

int main(int argc, char* argv[])
{
    if (strcmp(argv[1], "-e") == 0){
        if (argc < 5){
            cout<<"Error encode par!"<<endl;
            return 1;
        }
        unsigned long int ct = encode(str2int(argv[2]), argv[3], argv[4]);
        cout<<"Encoded "<<ct<<" bits"<<endl;
    }
    else if (strcmp(argv[1], "-d") == 0){
        if (argc < 6){
            cout<<"Error decode par!"<<endl;
            return 1;
        }
        decode(str2int(argv[3]), str2int(argv[2]) , argv[4], argv[5]);
    }
    else{
        cout<<"Usage:"<<endl;
        cout<<"  -e <#context> <infile> <compressedfile>"<<endl;
        cout<<"  -d <#context> <#bits> <outfile> <compressedfile>"<<endl;
    }
  return 0;
}
