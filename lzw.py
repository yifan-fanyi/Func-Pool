import sys
from sys import argv
from struct import *


data = 'acadadbadac'  

print('input: ', data)
# taking the input file and the number of bits from command line
# defining the maximum table size
# opening the input file
# reading the input file and storing the file data into data variable
                   

# Building and initializing the dictionary.
dictionary_size = 5               
dictionary = {'a':1, 'b':2,'c':3,'d':4}    
string = ""             # String is null.
compressed_data = []    # variable to store the compressed data.

# iterating through the input symbols.
# LZW Compression algorithm
for symbol in data:                     
    string_plus_symbol = string + symbol # get input symbol.
    if string_plus_symbol in dictionary: 
        string = string_plus_symbol
    else:
        compressed_data.append(dictionary[string])
        if(len(dictionary) <= 256):
            dictionary[string_plus_symbol] = dictionary_size
            dictionary_size += 1
        string = symbol

if string in dictionary:
    compressed_data.append(dictionary[string])

print('Compressed: ', compressed_data)
    
print('dictionary: ', dictionary)
print()


compressed_data = [1,5,3,1,3,4,2,1,4,9]
print('decode: ', compressed_data)
dictionary_size = 5  
dictionary = {1:'a', 2:'b',3:'c',4:'d'} 
next_code = 5
decompressed_data = ""
string = ""

# iterating through the codes.
# LZW Decompression algorithm
for code in compressed_data:
    if not (code in dictionary):
        dictionary[code] = string + (string[0])
    decompressed_data += dictionary[code]
    if not(len(string) == 0):
        dictionary[next_code] = string + (dictionary[code][0])
        next_code += 1
    string = dictionary[code]

# storing the decompressed string into a file.


print('Decompressed: ', decompressed_data)
print('dictionary: ', dictionary)
