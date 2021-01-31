# 2020.11.20
# @yifan
#
# HM on Kodak evaluation
#

import numpy as np
import matplotlib.pyplot as plt
import sys

#      [Qp, Bitrate     ,  Y-PSNR ,   U-PSNR  ,  V-PSNR  ,  YUV-PSNR]
evl = np.array([
                [ 0, 140571.2000,   92.5525,   82.0636,   81.4651,   84.9744],
                [ 1, 136290.5000,   87.4312,   75.5555,   75.3026,   79.4295],
                [ 2, 132404.9800,   81.6609,   69.0980,   68.9170,   73.1700],
                [ 3, 129380.2600,   76.0014,   65.3742,   65.3157,   69.3424],
                [ 4, 126841.2400,   69.2526,   63.3433,   63.3232,   66.2207],
                [ 5, 121412.3400,   62.9091,   60.4882,   60.4892,   61.8556],
                [ 6, 113286.6600,   59.1848,   57.3631,   57.4519,   58.4808],
                [ 7, 104876.1200,   57.0988,   55.4766,   55.5468,   56.4888],
                [ 8,  95434.3400,   55.1438,   54.0475,   54.1268,   54.7548],
                [ 9,  89099.5000,   53.9912,   53.3348,   53.4537,   53.7733],
                [10,  82802.0000,   52.9450,   52.6660,   52.8219,   52.8646],
                [11,  77060.8000,   51.9581,   52.0962,   52.2800,   52.0167],
                [12,  71530.4800,   50.9703,   51.5656,   51.7736,   51.1702],
                [13,  66008.2600,   49.9758,   51.0235,   51.2451,   50.3032],
                [14,  60823.1200,   48.9913,   50.5128,   50.7522,   49.4432],
                [15,  55813.1400,   48.0028,   50.0366,   50.2725,   48.5761],
                [16,  51220.1600,   47.0527,   49.4584,   49.6784,   47.7004],
                [17,  46819.9600,   46.1483,   48.9930,   49.1990,   46.8793],
                [18,  43008.8000,   45.3441,   48.5491,   48.7727,   46.1403],
                [19,  39288.5800,   44.5193,   48.0628,   48.2590,   45.3674],
                [20,  35814.9200,   43.7062,   47.5758,   47.7702,   44.6030],
                [21,  32819.4200,   42.9666,   47.1027,   47.2998,   43.8973],
                [22,  29983.1200,   42.2176,   46.6066,   46.7925,   43.1735],
                [23,  27289.4000,   41.4610,   46.1239,   46.2798,   42.4378],
                [24,  24890.7200,   40.7463,   45.6068,   45.7514,   41.7333],
                [25,  22508.6400,   39.9725,   45.1253,   45.2373,   40.9778],
                [26,  20192.2600,   39.1985,   44.6278,   44.6937,   40.2134],
                [27,  18174.7200,   38.4813,   44.1229,   44.1837,   39.4997],
                [28,  16272.2600,   37.7587,   43.6521,   43.6744,   38.7805],
                [29,  14463.4400,   37.0188,   43.0688,   43.1345,   38.0363],
                [30,  12993.5200,   36.3369,   43.0966,   43.1861,   37.4053],
                [31,  11445.9600,   35.6371,   42.6034,   42.6113,   36.6963],
                [32,  10008.5000,   34.9285,   42.0717,   42.1183,   35.9786],
                [33,   8764.8200,   34.2605,   41.5569,   41.6031,   35.2972],
                [34,   7618.8000,   33.6076,   41.0465,   41.1007,   34.6310],
                [35,   6644.0200,   32.9683,   41.1032,   41.1574,   34.0186],
                [36,   5751.0000,   32.3747,   40.5536,   40.6671,   33.4044],
                [37,   4971.7200,   31.7682,   40.5691,   40.6901,   32.8164],
                [38,   4231.1600,   31.1740,   40.1267,   40.2405,   32.2014],
                [39,   3648.0000,   30.6152,   40.1597,   40.2484,   31.6613],
                [40,   3091.3200,   30.0744,   39.7724,   39.8051,   31.1021],
                [41,   2645.2000,   29.5531,   39.7495,   39.8369,   30.5843],
                [42,   2239.1200,   29.0659,   39.3347,   39.3731,   30.0804],
                [43,   1913.8800,   28.5739,   39.4119,   39.4202,   29.5965],
                [44,   1598.5200,   28.0944,   38.8745,   38.9796,   29.1031],
                [45,   1341.5800,   27.6428,   38.4699,   38.4781,   28.6411],
                [46,   1124.5200,   27.1800,   38.0606,   38.1044,   28.1801],
                [47,    937.9000,   26.7286,   37.6180,   37.6193,   27.7281],
                [48,    788.2800,   26.3194,   37.2279,   37.2711,   27.3189],
                [49,    658.3600,   25.9207,   36.8089,   36.8027,   26.9210],
                [50,    552.6000,   25.5335,   36.4018,   36.3489,   26.5382],
                [51,    468.5400,   25.1808,   35.8953,   35.8766,   26.1826]])

def bpp_psnr():
    plt.plot(evl[:, 1], evl[:, -1], label='Kodak', c='b')
    plt.xlabel("Bitrate")
    plt.ylabel('PSNR')

def qp_psnr():
    plt.plot(evl[:, 1], evl[:, -1], c='b', label='Kodak')
    plt.xlabel("Qp")
    plt.ylabel('PSNR')

def show():
    plt.legend()
    plt.show()

def plt_point(x, y):
    plt.scatter(x, y, label='Our', c='r')


bpp_psnr()
if len(sys.argv) > 1:
    plt_point(float(sys.argv[1]), float(sys.argv[2]))
show()

'''
# 51
    POC    0 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6424 bits [Y 24.7413 dB    U 36.4685 dB    V 34.5766 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bd910e220195dd979ac244f15f327257,35277bfd98ae4a09b47d83d176e36d21,4e5090c98cc1f181630d3701c70c2d5c]
    POC    1 TId: 0 ( I-SLICE, nQP 51 QP 51 )      16224 bits [Y 21.6038 dB    U 32.8480 dB    V 32.9411 dB] [ET     2 ] [L0 ] [L1 ] [MD5:57bb5172e80ac855c1e2ed82da2fcb3f,45b9093e61b3f03a92b50a17b56cecd0,08bcc178e822ca5cb693b4cddd703c32]
    POC    2 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6176 bits [Y 27.4010 dB    U 38.4104 dB    V 32.9431 dB] [ET     2 ] [L0 ] [L1 ] [MD5:64a44940a3cea995d94c6a6ac04de95c,1321e2476b55ec0d781c974db7fbffb6,8323745a31cb39323400dadbd0b0c93b]
    POC    3 TId: 0 ( I-SLICE, nQP 51 QP 51 )       5968 bits [Y 26.1656 dB    U 36.9247 dB    V 37.4571 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ca986e24295b7bdc869a8561b174681f,3cb77dee76c2fed63e965d711c0075e7,9921b78bad55c9d255e6d51d01460368]
    POC    4 TId: 0 ( I-SLICE, nQP 51 QP 51 )       5600 bits [Y 23.8396 dB    U 36.9150 dB    V 37.3522 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7b800bf6614236b94ea0b8875d1144e7,c14f7e31b615a077d2fdff375b008548,48d1c839302ffbc60c5b9a72c63e1460]
    POC    5 TId: 0 ( I-SLICE, nQP 51 QP 51 )       4816 bits [Y 27.7531 dB    U 37.5528 dB    V 36.7879 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4a675e7a53c4f62b6df7ff72ec00e8bb,84c7e5043c11d200d0938bcd6a2a1657,255d8fa0ec8b2558910cb7c7083f158a]
    POC    6 TId: 0 ( I-SLICE, nQP 51 QP 51 )      10952 bits [Y 20.7810 dB    U 35.2781 dB    V 37.9391 dB] [ET     2 ] [L0 ] [L1 ] [MD5:325203faac2c7a95225d5e73fdb2e255,12a1f9edab72a803fd99d905a4ea380d,818279b559c6c6e4c00b8d0151c3316e]
    POC    7 TId: 0 ( I-SLICE, nQP 51 QP 51 )       8824 bits [Y 25.5745 dB    U 33.9771 dB    V 35.5720 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f09b7b1e6997a16d755363cddfb2668d,8df301431077fc3a408105e19d264e9b,c78a9321d4869c3981dccd63aaa54e84]
    POC    8 TId: 0 ( I-SLICE, nQP 51 QP 51 )       4928 bits [Y 27.9543 dB    U 35.8930 dB    V 36.4851 dB] [ET     2 ] [L0 ] [L1 ] [MD5:83d605fb56b70c29c54eed805c6bc5e8,a8c1edc107093377c9a90f41caab145f,cb81f8d3ad8fa26bbd32734038521224]
    POC    9 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6728 bits [Y 25.7894 dB    U 37.4826 dB    V 39.0117 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e47fedf7d17855c9e7e3224930799453,a7c20efc6c2676f83ac5f4005cd79b62,3a448e6ed58db89bcf4876d458a2d326]
    POC   10 TId: 0 ( I-SLICE, nQP 51 QP 51 )       3872 bits [Y 26.7367 dB    U 37.7930 dB    V 40.2892 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e4d35c74c34619952dd186a1352bbc20,fd0343efd2bc4e67d4fafe28958c3fa0,a6bc6b396d181bf63e9a8635406a06e5]
    POC   11 TId: 0 ( I-SLICE, nQP 51 QP 51 )       3392 bits [Y 28.2836 dB    U 35.3531 dB    V 31.8109 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3be34b256aa490be1389cb4bea1ccf8d,2c0ee43eed00559cac84f76540a87e3b,e71dacd16a71584f38ffb9d3fb931803]
    POC   12 TId: 0 ( I-SLICE, nQP 51 QP 51 )      10632 bits [Y 23.7830 dB    U 32.9642 dB    V 32.2382 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b1f1c804e75e16c9274d19f576ca5f04,c30dd7e24c7de71b11e5364f514c4da3,09a07b31eb138d7fbc30cf5ea7b06a87]
    POC   13 TId: 0 ( I-SLICE, nQP 51 QP 51 )       8240 bits [Y 22.5880 dB    U 37.6085 dB    V 35.8013 dB] [ET     2 ] [L0 ] [L1 ] [MD5:828ed7c7244f3aad3c7cf3d0cca577e6,8367f82d0993dbb5ae6e1c35635f9c0c,6d5cacede53b276fbe85810eb0c231ed]
    POC   14 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6360 bits [Y 27.0573 dB    U 36.3228 dB    V 33.9236 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b3c54510799da723a3b93e937949913a,bde5a1c1c676953ab1a93674e5b6bf38,cb2fec5f7d024c0e47f73ed9ebf88dcd]
    POC   15 TId: 0 ( I-SLICE, nQP 51 QP 51 )       8448 bits [Y 23.0593 dB    U 34.7852 dB    V 36.0116 dB] [ET     2 ] [L0 ] [L1 ] [MD5:33ccd8af0fa115c66fceabd9f5c08aef,aa5e3cc18dd4c532d5e98f594951f709,94066eb823f09c8d86090b178a175074]
    POC   16 TId: 0 ( I-SLICE, nQP 51 QP 51 )       7784 bits [Y 23.5031 dB    U 34.4129 dB    V 35.2862 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c7721db907291f5dfa11e0054625d1dd,901a5284fc4b44dbbda35c8cb78b51c9,d2479fee62fb937b7cf7bb85ac3192ea]
    POC   17 TId: 0 ( I-SLICE, nQP 51 QP 51 )       7976 bits [Y 25.4235 dB    U 37.4145 dB    V 37.4736 dB] [ET     2 ] [L0 ] [L1 ] [MD5:752eecae5a316cbfd5dc17f2af94ad98,068048593e1680031ffbe630bd00e9ce,8d00d1842354a5ef2f6b42f170837d1e]
    POC   18 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6016 bits [Y 25.7507 dB    U 35.1336 dB    V 35.2798 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fd6d5f2b421544f2b17b727458def8b6,d4ecc9766c675eea3d2953390fa05f2b,5b28831fe11bde374dd802a6edc8d00c]
    POC   19 TId: 0 ( I-SLICE, nQP 51 QP 51 )       7160 bits [Y 28.1708 dB    U 35.0899 dB    V 34.7641 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2c35cd6bf7e7ea583de803dc9fd98f9f,24533fcbdc0a438556818295b7e35579,f3babb93e63c01941106aa31fa148232]
    POC   20 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6656 bits [Y 26.5651 dB    U 36.4961 dB    V 37.7347 dB] [ET     2 ] [L0 ] [L1 ] [MD5:930f34c5037711dc1538ac01e234da25,0355a669183fc150d323e49239fb1f38,49ea8bd98918eb993a6f97fa3d4be078]
    POC   21 TId: 0 ( I-SLICE, nQP 51 QP 51 )       8312 bits [Y 24.1993 dB    U 35.9209 dB    V 36.9107 dB] [ET     2 ] [L0 ] [L1 ] [MD5:756922d786786b3a43ce914f0cce4703,09c237e560d51fae6e293fb96a76b98f,fa99f8a237c2cd55c8c9206586d64d0d]
    POC   22 TId: 0 ( I-SLICE, nQP 51 QP 51 )       6240 bits [Y 26.4327 dB    U 36.1270 dB    V 38.9656 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c22431641e20742a04e9b165c077dd79,80bfe43f43a397660c7a4cbb00f8ffaa,83dcb9b4664af253beeb019d9d55acf0]
    POC   23 TId: 0 ( I-SLICE, nQP 51 QP 51 )      19688 bits [Y 21.1831 dB    U 34.3143 dB    V 33.4832 dB] [ET     2 ] [L0 ] [L1 ] [MD5:11ea039737fb8beebf511c00505d277e,3a9886b05b1638c7ce419d506c46d70d,289d2d4a1fa3d8fcfacc4fd754e38dc4]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a     468.5400   25.1808   35.8953   35.8766   26.1826  

# 50
    POC    0 TId: 0 ( I-SLICE, nQP 50 QP 50 )       7728 bits [Y 25.0996 dB    U 37.1101 dB    V 35.1991 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fa42c046532aa4ad03e64ce9801f1ce6,4635303bb26a135c916ffb94797ff294,f303d5ed621028c6a6fc3854d719181b]
    POC    1 TId: 0 ( I-SLICE, nQP 50 QP 50 )      19616 bits [Y 21.9799 dB    U 33.2215 dB    V 33.4277 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e1e78e9a8a029392f1b6f640c02cf374,e098d9c13a512f5ada3d9311775ac068,053e4723ce65694c3e5cff822f39a36a]
    POC    2 TId: 0 ( I-SLICE, nQP 50 QP 50 )       6952 bits [Y 27.6990 dB    U 38.4802 dB    V 33.5383 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ab5d0651b7744999e0c3205f13944220,b6ff22bc700a7ad70e1313845075698e,5f052db705c9db568a6627723d24e533]
    POC    3 TId: 0 ( I-SLICE, nQP 50 QP 50 )       7040 bits [Y 26.6940 dB    U 37.4473 dB    V 37.7375 dB] [ET     2 ] [L0 ] [L1 ] [MD5:969e8f7775e8bb97dcdfc3cdf37db755,bcc9e04ec15e37e335daf120e03cedb2,22391bd5805b6aebbbd07cfd6d80fcfa]
    POC    4 TId: 0 ( I-SLICE, nQP 50 QP 50 )       6824 bits [Y 24.1868 dB    U 37.3879 dB    V 37.9700 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8a88ca1e09b080a99c6b005d6c6d1f73,0b69cfab24349fc1e41e5ff7134f1e79,1d73d4329932730ffc3bbe043160562f]
    POC    5 TId: 0 ( I-SLICE, nQP 50 QP 50 )       5216 bits [Y 28.0902 dB    U 38.1846 dB    V 37.0141 dB] [ET     2 ] [L0 ] [L1 ] [MD5:581ad987cdcac4522b7d31deb31360da,0839c51754fd42ab5fcbd8e9824a1305,6e6f26b3043866cabb33c64a6abff7ac]
    POC    6 TId: 0 ( I-SLICE, nQP 50 QP 50 )      13544 bits [Y 21.0436 dB    U 35.5903 dB    V 37.8506 dB] [ET     2 ] [L0 ] [L1 ] [MD5:388a4b52ad0706a8fdeee72b52c81f0b,79fe08371d0e78e24020b8f134f93693,f6a3e1d18e84323b810efe26e8c49e89]
    POC    7 TId: 0 ( I-SLICE, nQP 50 QP 50 )      10592 bits [Y 26.1151 dB    U 34.3463 dB    V 36.3737 dB] [ET     2 ] [L0 ] [L1 ] [MD5:05eba83eea642eece138578e1c7bd62f,9fdf29752125bba83c2b151d6c3b8d2f,ace7ee2f877af5d4cde171efa2dfaedc]
    POC    8 TId: 0 ( I-SLICE, nQP 50 QP 50 )       5712 bits [Y 28.2594 dB    U 36.5866 dB    V 37.3108 dB] [ET     2 ] [L0 ] [L1 ] [MD5:92dbac8c863efb1598b48bdf72b99998,7d3cedb3ab30d21196dc6372d3413dbb,84d869a4095b8883afbae92a4f4dfa86]
    POC    9 TId: 0 ( I-SLICE, nQP 50 QP 50 )       8104 bits [Y 26.1984 dB    U 38.5577 dB    V 39.3403 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c5b9d5d0f8b6a881062e1400cf16c392,9101f530dfa095fe577f98f1ced9ae1d,b76688d60a1addb4c0172db8569c58d8]
    POC   10 TId: 0 ( I-SLICE, nQP 50 QP 50 )       4184 bits [Y 26.9189 dB    U 38.9065 dB    V 40.5456 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3351c80e6d3c1a54eab7f76406ed43f2,0fc235ce6646998519231b900dec0c09,fb166912fa31f4c1018758ccb607f5ee]
    POC   11 TId: 0 ( I-SLICE, nQP 50 QP 50 )       3968 bits [Y 28.5021 dB    U 35.8276 dB    V 32.8124 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c2d8437f166158ece13cd9bbbb4406d7,85d5cb78c2652fce4d62797f0fab5bce,1e178dc0b1acdb52044df7e990dd607c]
    POC   12 TId: 0 ( I-SLICE, nQP 50 QP 50 )      12416 bits [Y 24.0895 dB    U 33.4577 dB    V 32.9493 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dfc04629f489ac1d3bf8221b9b9f56f3,f8c4b7f2a8a3ce1f5e55c22dc3fc3886,bdc1e870ba8b2c014886a149d120a061]
    POC   13 TId: 0 ( I-SLICE, nQP 50 QP 50 )       9952 bits [Y 22.8984 dB    U 38.1263 dB    V 36.2266 dB] [ET     2 ] [L0 ] [L1 ] [MD5:307f6bc95dc15ef2aa40eae1bbf50f71,98efb904ac7cb5ad03358995d9d2b89f,869ba2361d5a8cead903543a5ae7bb62]
    POC   14 TId: 0 ( I-SLICE, nQP 50 QP 50 )       7336 bits [Y 27.4661 dB    U 37.3422 dB    V 34.3445 dB] [ET     2 ] [L0 ] [L1 ] [MD5:44ddf05a4353b20e14494a1230e3fad5,b0b1ccaef72efd309c76c51d63e89636,86a3eb2d119a36994f27265c37f865ac]
    POC   15 TId: 0 ( I-SLICE, nQP 50 QP 50 )      10080 bits [Y 23.3325 dB    U 34.9970 dB    V 36.6421 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fbf1e019d2f01bce77a93072e08b33c2,17ddfd381b9a1cedbf7fac6635a420bb,1b4c4eb478ffb785a89c0860c2cdd32b]
    POC   16 TId: 0 ( I-SLICE, nQP 50 QP 50 )       9416 bits [Y 23.7766 dB    U 34.8567 dB    V 35.6684 dB] [ET     2 ] [L0 ] [L1 ] [MD5:581cd010951d1b8ca82d4cce3f32c3db,cc1af3edeaf9d56b39ca01f58ba382d6,b120e7883e091f68773d17f63116cb4b]
    POC   17 TId: 0 ( I-SLICE, nQP 50 QP 50 )       9336 bits [Y 25.8857 dB    U 37.3783 dB    V 38.1336 dB] [ET     2 ] [L0 ] [L1 ] [MD5:417b32b47019cdf81460ab9ca79f0985,3a6b628817cfefc5341fdffae2a72ea0,c95bcafcff492280aa9853ce8bedb1e9]
    POC   18 TId: 0 ( I-SLICE, nQP 50 QP 50 )       6912 bits [Y 26.0003 dB    U 35.5116 dB    V 35.8314 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1a9ba39a4b9c1a7c6895371cff351769,1f8f1b92de7a62b3e8945865bbff99f8,da7a4f005df4e386188b1759a502280d]
    POC   19 TId: 0 ( I-SLICE, nQP 50 QP 50 )       8040 bits [Y 28.6034 dB    U 35.4608 dB    V 35.3624 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fcdc395230b8614d5cf5fe2cd0718383,54f9e04b6f320f3d8ffd5f0c44cd515a,3b5884ae8f38cd30d57d6e33b9206324]
    POC   20 TId: 0 ( I-SLICE, nQP 50 QP 50 )       7488 bits [Y 26.9528 dB    U 37.3832 dB    V 37.9261 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6ffba61d2df0e5e2fd47a07b8f6fe649,0deb0379d65b6abad1236197c5d483d6,1c431c9d923bead28264a35f55ac6b83]
    POC   21 TId: 0 ( I-SLICE, nQP 50 QP 50 )       9888 bits [Y 24.5132 dB    U 36.6181 dB    V 37.3190 dB] [ET     2 ] [L0 ] [L1 ] [MD5:54fd75b2e2a5cb4f9330293c44f8b235,fc0afe0ba51f39bb39f7995c6b9e6b1e,b871f4df0334bafa5af3f11d1f74e2c6]
    POC   22 TId: 0 ( I-SLICE, nQP 50 QP 50 )       7136 bits [Y 26.8398 dB    U 36.1700 dB    V 39.1412 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e64acaf95dc67f30461609a77ffc5719,6940f1e49324b19b6c79fbbc9c3e0094,79b26a75bf6c7bae382b19f9e4184e46]
    POC   23 TId: 0 ( I-SLICE, nQP 50 QP 50 )      23560 bits [Y 21.6587 dB    U 34.6948 dB    V 33.7085 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dbba1a13dafe9b95934e97789ef83ffc,2e0306f3fd5ba013e70d16874305db15,d0e2a3b97a6d980cea133a724c63682e]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a     552.6000   25.5335   36.4018   36.3489   26.5382  

# 49
    POC    0 TId: 0 ( I-SLICE, nQP 49 QP 49 )       9400 bits [Y 25.4968 dB    U 37.6760 dB    V 35.5579 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4fdeb4447c385ec187bb3a4d44c9d6b1,d5df0ab36dcff35b13ebe7cc6c3e82b1,30a86c16021d331db30f40db5d1491bb]
    POC    1 TId: 0 ( I-SLICE, nQP 49 QP 49 )      23880 bits [Y 22.4543 dB    U 33.6348 dB    V 33.6485 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a0b4f13a59901973d2985cecfa798f58,39a8f0ff7c2c0f823502079e62d3a792,293b17708c81dd4c0fdd836abf51a01c]
    POC    2 TId: 0 ( I-SLICE, nQP 49 QP 49 )       7664 bits [Y 28.0034 dB    U 38.9963 dB    V 33.8736 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bed76caa9af3bc93c0e8f816b281a27d,602e38f5656ffea34b9dbc2970278e5e,cacf29de713b37c5310412acc187d0ee]
    POC    3 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8376 bits [Y 27.0670 dB    U 37.6925 dB    V 38.2794 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0a074ec0fe2500cd13a668a679776108,9a313fdfe5ecbde70219e1782df13de6,d4452c4c60d0334b68f427708fa1cb48]
    POC    4 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8416 bits [Y 24.4850 dB    U 37.7049 dB    V 38.2197 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d254d7446e065dacfeee17475ac21952,27b9a131aab67e5c70a343a7b5d84217,b5fef37d16e227057aa55da1e1e6d05f]
    POC    5 TId: 0 ( I-SLICE, nQP 49 QP 49 )       6080 bits [Y 28.4175 dB    U 38.7815 dB    V 37.9580 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a696a39cf3270f243336695de6963c3d,2b5551ac0a3dfa6ac7e76ff98704b7a8,3f833814b0d19489d69a096d98cec935]
    POC    6 TId: 0 ( I-SLICE, nQP 49 QP 49 )      17560 bits [Y 21.3616 dB    U 35.7249 dB    V 38.2395 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fb707f6a1271b317aca62b02fcb69f0c,8a45503a53ef3d030a94dbc080765135,87c3ae298bfbf8a67b14a88d9d8ebab2]
    POC    7 TId: 0 ( I-SLICE, nQP 49 QP 49 )      12328 bits [Y 26.5890 dB    U 34.7987 dB    V 36.7831 dB] [ET     2 ] [L0 ] [L1 ] [MD5:64c2461425656879480579d3c39a5988,11fa98b22bcd32e187d08d1580983aa4,8dfc017341038cc1b273437412eb587d]
    POC    8 TId: 0 ( I-SLICE, nQP 49 QP 49 )       6584 bits [Y 28.8048 dB    U 36.9841 dB    V 37.9101 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5ca04733ff2b93e8579fb263d5ad7ef4,5f13edd938ad0d5b15664fce67811759,6e2fd99fcfbb11a17ceede6c9154dd79]
    POC    9 TId: 0 ( I-SLICE, nQP 49 QP 49 )       9544 bits [Y 26.6407 dB    U 38.7961 dB    V 39.6780 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d3c42d2fade6d98a0c6e708fa1f3f5b8,38bfdb691d88e0ad9d1afed3be15f555,9d90adbc7868dcb5d588a773e10b2b86]
    POC   10 TId: 0 ( I-SLICE, nQP 49 QP 49 )       5248 bits [Y 27.1937 dB    U 39.0179 dB    V 40.9746 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4385d52df81adde73e73b3dcddc9b9da,e9aae03cc5a1dd2e8e8f0c3f1acaf228,4f532f6b7e1805b922a0443b967c98a0]
    POC   11 TId: 0 ( I-SLICE, nQP 49 QP 49 )       4672 bits [Y 28.7515 dB    U 36.2946 dB    V 33.1306 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b99c6b7d85041e1d5febf6d6d9a599b2,aa94352c85a3fead7cf67724444e70c8,b84562e7ee8b4980b753c15900338007]
    POC   12 TId: 0 ( I-SLICE, nQP 49 QP 49 )      14792 bits [Y 24.3799 dB    U 33.8905 dB    V 33.7242 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4ee4a7822fbc00418b7c467f8fc3a592,175c8e52cf13d0c42cd7a14577f9bdb1,5ce55dc6cfa8fbd8570e45361095a3f1]
    POC   13 TId: 0 ( I-SLICE, nQP 49 QP 49 )      12096 bits [Y 23.2391 dB    U 38.4082 dB    V 36.1738 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c407cbc71cf3920726d3e2a4c68e5128,cdf7d2e559e183316232d2e8c99fd5fb,aef1670e6906ed01476d504b81306ef9]
    POC   14 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8288 bits [Y 27.9750 dB    U 37.7121 dB    V 34.8771 dB] [ET     2 ] [L0 ] [L1 ] [MD5:392ff65a5738df49d7a768def016b2ec,4d45732815f8cd0e98e3e5f0c5a35e68,7afd1b1e2434a86a3891a82daeb80bfd]
    POC   15 TId: 0 ( I-SLICE, nQP 49 QP 49 )      12432 bits [Y 23.6570 dB    U 35.5121 dB    V 36.6498 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e5cb122acb6edb233c0260d73e4c04aa,b5ce764d584a14ca1ee3480e4b156ad9,415852f3dba982fd831389c189447c9c]
    POC   16 TId: 0 ( I-SLICE, nQP 49 QP 49 )      11368 bits [Y 24.0428 dB    U 35.2528 dB    V 36.0990 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5504e34f4b206b18f745d3d8d3ebc23f,a05982b8c044b3c95cf00995d6e5e73c,8efe175af3a77612f1931834de4030d0]
    POC   17 TId: 0 ( I-SLICE, nQP 49 QP 49 )      10776 bits [Y 26.3075 dB    U 37.9889 dB    V 38.3226 dB] [ET     2 ] [L0 ] [L1 ] [MD5:62382a4aaa0fd063612bd972f4b10222,28552d08e6cf27844d39c2fb5c873794,3e79f3fa58533852a7e671bb49f7f3d3]
    POC   18 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8032 bits [Y 26.3111 dB    U 35.8819 dB    V 36.0903 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f15e9d860e8794d5aae5a4081ebe81a3,167bce7136b7da5066e85810b2b30387,b023b6e31814e548e32e3855e1070aeb]
    POC   19 TId: 0 ( I-SLICE, nQP 49 QP 49 )       9208 bits [Y 29.1146 dB    U 35.9344 dB    V 35.8494 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a45d01dc9e80f512c405bdc3b89663a7,60db600f31db729c8003ed4cf9af3e47,68a011e927c8dc311f593279ef056464]
    POC   20 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8600 bits [Y 27.4833 dB    U 37.7557 dB    V 39.0755 dB] [ET     2 ] [L0 ] [L1 ] [MD5:755386b8e37a3c188ca3610efde63deb,be18fe192517d4f20f3f5f7bd8371995,2d1e7dc0c4958d516d1311d4ef793194]
    POC   21 TId: 0 ( I-SLICE, nQP 49 QP 49 )      12144 bits [Y 24.9130 dB    U 36.8338 dB    V 38.1855 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3a7ea088d7cbd59381b8a3ad224bfa7b,fe8ba94c806bc2c2ff846d51ab232aed,099eb2e2707a1d2e65a8c75da8cb2c2d]
    POC   22 TId: 0 ( I-SLICE, nQP 49 QP 49 )       8152 bits [Y 27.2526 dB    U 36.9767 dB    V 39.7440 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1261aaf491ae2af3b1f709e5db4a13c0,2bdbfca26c9be80d6c9fa7a51c72c559,af80b8c5215268f0585abed6ece41c9e]
    POC   23 TId: 0 ( I-SLICE, nQP 49 QP 49 )      27704 bits [Y 22.1547 dB    U 35.1651 dB    V 34.2198 dB] [ET     2 ] [L0 ] [L1 ] [MD5:63c71e45c502202206e5b5a23da2e4e8,d24378bce019661203e83c34b896a712,97ff2abb6b4a3d59cb29d361d0e661d4]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a     658.3600   25.9207   36.8089   36.8027   26.9210  

# 48
    POC    0 TId: 0 ( I-SLICE, nQP 48 QP 48 )      11160 bits [Y 25.8175 dB    U 38.0753 dB    V 35.9564 dB] [ET     3 ] [L0 ] [L1 ] [MD5:49d23c1a142e48f21d9fe1cc33ad38c5,a937f4511e4ccecd77de2bc46c12120f,302cc781c52ebdf6f0547eb661506e11]
    POC    1 TId: 0 ( I-SLICE, nQP 48 QP 48 )      28408 bits [Y 22.9312 dB    U 34.0159 dB    V 34.1189 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3588579ffb9b6c17f667d56f6631c42b,0f2fa508117ffc33cdeaddf36db52717,8b09929cf2f0526f63ccb230eb4f446b]
    POC    2 TId: 0 ( I-SLICE, nQP 48 QP 48 )       9480 bits [Y 28.4486 dB    U 39.5950 dB    V 34.3890 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ee085043f414c7ae7ee95a69434f8efa,38934c2970316b1857464d856a1ac09e,15c377a62881b9f81d85e24dfaaf6aac]
    POC    3 TId: 0 ( I-SLICE, nQP 48 QP 48 )      10264 bits [Y 27.6659 dB    U 38.0374 dB    V 38.7018 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f67f702bda4abc0d412566da4bcf37a1,f9734e3833ed9aa47d897dd3ff6a6cd4,d424bb1ba61ae55280ad8da8d723c750]
    POC    4 TId: 0 ( I-SLICE, nQP 48 QP 48 )      10616 bits [Y 24.8303 dB    U 38.0845 dB    V 38.9354 dB] [ET     2 ] [L0 ] [L1 ] [MD5:60ce35879715d831c1fdc0b4360cad1b,6e6db25692d0fc45e7dcda79a89488e9,ff328325779f943bb5b7e80d0c5bbf43]
    POC    5 TId: 0 ( I-SLICE, nQP 48 QP 48 )       6824 bits [Y 28.7834 dB    U 39.3024 dB    V 38.5368 dB] [ET     2 ] [L0 ] [L1 ] [MD5:893d1a2ca48990422f617baf2b0ec9e6,15b2be2435f19876a22c985c9f18502f,036b78d3cbac775795d1d12a9abcf1ed]
    POC    6 TId: 0 ( I-SLICE, nQP 48 QP 48 )      23160 bits [Y 21.7455 dB    U 35.9621 dB    V 38.5580 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9871eda649060ecf17fa9851a947f950,901b241aadf5b96e0c4eced65bd10a9f,ad6b2e0305098288a21b4b4c9e7ea7db]
    POC    7 TId: 0 ( I-SLICE, nQP 48 QP 48 )      14192 bits [Y 27.0916 dB    U 35.0210 dB    V 37.2994 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1cfe9ca44f37974a8edb6f1d749012f0,62a1fb69e29fcefa310d63ebd2941179,24104434585503097458927ec45cc5a6]
    POC    8 TId: 0 ( I-SLICE, nQP 48 QP 48 )       7408 bits [Y 29.1540 dB    U 37.3958 dB    V 38.3770 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b7a324c53de84027c239df6b58fade26,d38d1d563eb24ec44eed7c684c498383,f7f6f2affa73cad4dde0dc95b81633e9]
    POC    9 TId: 0 ( I-SLICE, nQP 48 QP 48 )      11024 bits [Y 27.0043 dB    U 39.3062 dB    V 39.8300 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8a1857d2bf3db11f62025edb24e65fd3,9cc4ec13908f72a7840e384e0b75dc5d,5ba2065e232f1e7f064d6deb002e334d]
    POC   10 TId: 0 ( I-SLICE, nQP 48 QP 48 )       5952 bits [Y 27.3690 dB    U 39.6464 dB    V 41.3050 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cab93830a18c590eb0eb0f815511e39a,9c2cc7386ba85c5d79d433cfac8d54d4,cde27db704f83acf1334740674499d66]
    POC   11 TId: 0 ( I-SLICE, nQP 48 QP 48 )       5552 bits [Y 28.9682 dB    U 36.8831 dB    V 34.0898 dB] [ET     2 ] [L0 ] [L1 ] [MD5:96803e562a5cb2132693d0f5783a01cc,e3dd0ea978014f0da5843b9fc18dda06,f5ce80245f4848d178a13f2802242f3c]
    POC   12 TId: 0 ( I-SLICE, nQP 48 QP 48 )      18216 bits [Y 24.7561 dB    U 34.4481 dB    V 34.2563 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d04b0004b5ff85609e653f75d730a7aa,2a3677297ac1927ab4e89d87b69b7fac,0f95e995d85b58dbf4b0735daf6ac73d]
    POC   13 TId: 0 ( I-SLICE, nQP 48 QP 48 )      14536 bits [Y 23.5552 dB    U 39.0502 dB    V 36.4642 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8c8157a8b88709f1ee3a45cbf71f85c0,23c53615cf06c7c85cd55fe32ab210ab,9b9d0c0a0ddeb32835e1b76743216ff4]
    POC   14 TId: 0 ( I-SLICE, nQP 48 QP 48 )       9088 bits [Y 28.2688 dB    U 38.0544 dB    V 35.4525 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3856431c266f1ccb0df3840294df9d38,508502bbd54f4d3c4a8e62c703907b7c,66a48fea67a6748bea110a39562dfb4d]
    POC   15 TId: 0 ( I-SLICE, nQP 48 QP 48 )      15696 bits [Y 24.0303 dB    U 35.7814 dB    V 37.2217 dB] [ET     2 ] [L0 ] [L1 ] [MD5:95f9ff4545e3d1bdc92c815dd96dfc9e,267368644d537c20c15928bd240578e7,f2b2dd07fa07f5a8299569d89f3d76cf]
    POC   16 TId: 0 ( I-SLICE, nQP 48 QP 48 )      13784 bits [Y 24.3273 dB    U 35.5751 dB    V 36.3875 dB] [ET     2 ] [L0 ] [L1 ] [MD5:44725a03f5020b9b4ea874a63eda43da,9ce38ac5bf2ebd456377b6f995db54a5,2a7afe60f5c03907b5704962ef2df868]
    POC   17 TId: 0 ( I-SLICE, nQP 48 QP 48 )      12848 bits [Y 26.7206 dB    U 38.3074 dB    V 39.0444 dB] [ET     2 ] [L0 ] [L1 ] [MD5:62cc8171895cd752142916985107dbb8,aa74805f83d40eb9e924d9daca82441d,46c75bc908473e2dfa544529d8432925]
    POC   18 TId: 0 ( I-SLICE, nQP 48 QP 48 )       9648 bits [Y 26.6470 dB    U 36.0985 dB    V 36.6335 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2bcb51e860761dacd31b724ed0901a76,1906913cf58aa9892a8ac5798b364313,7670226842817d62b1d46b23a058e878]
    POC   19 TId: 0 ( I-SLICE, nQP 48 QP 48 )      10528 bits [Y 29.6966 dB    U 36.4549 dB    V 36.3357 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1186bfd6fa4392818709fd5f2c192013,abd0c3a64a46fa0eb88f89850b5fc8a7,909a87be209501c8f5059fcb6cfe86b8]
    POC   20 TId: 0 ( I-SLICE, nQP 48 QP 48 )      10024 bits [Y 27.9471 dB    U 38.1825 dB    V 39.3392 dB] [ET     2 ] [L0 ] [L1 ] [MD5:69544199bac6bffc963dbadcd8fd258e,6b975aba9321ff1c93bd3791c7068ca4,97f8c98213f12d9332ecd06996d30916]
    POC   21 TId: 0 ( I-SLICE, nQP 48 QP 48 )      14048 bits [Y 25.2687 dB    U 37.0203 dB    V 38.6895 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b7b1cf9a0f715d3a8233d97b9a279350,eb345309bab5a5721e35fb47937632c6,2497e4959a4efa31fb779933b701d655]
    POC   22 TId: 0 ( I-SLICE, nQP 48 QP 48 )       9608 bits [Y 27.9886 dB    U 37.4696 dB    V 39.8198 dB] [ET     2 ] [L0 ] [L1 ] [MD5:063253fd6cf2879aa4793853a4d784df,c31e5a33483d0a4cc7c9d6f901630afc,79b6187b92de91e4e59f1c49f2eb5401]
    POC   23 TId: 0 ( I-SLICE, nQP 48 QP 48 )      33248 bits [Y 22.6486 dB    U 35.7027 dB    V 34.7656 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6c3f0ee5299321957da9a8edc5c34240,bfc7fe17ac06e631a53f3b3bdf07ad74,4c1c4e7ca8dd4717448711b6a0a88210]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a     788.2800   26.3194   37.2279   37.2711   27.3189  

# 47
    POC    0 TId: 0 ( I-SLICE, nQP 47 QP 47 )      13344 bits [Y 26.1972 dB    U 38.5941 dB    V 36.3333 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5a7ecad56787931d3e1d34d609d236ff,022db2483636df24cc0cae165c9040ad,3d157af77869a5333c6b88bb4e3b9a44]
    POC    1 TId: 0 ( I-SLICE, nQP 47 QP 47 )      33392 bits [Y 23.3716 dB    U 34.4121 dB    V 34.4463 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ad876b8e0dff840cb38b1c7a691d917f,1e1fb04253dbd2cff4db9a9579675eec,e14005dc7f358a0a06e273d9ca85efb2]
    POC    2 TId: 0 ( I-SLICE, nQP 47 QP 47 )      10520 bits [Y 28.7808 dB    U 39.7923 dB    V 34.7409 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5caae1516c3d4adbadf36b1af9b3ea66,cc6eae1c691157f87abe46b47df7f50d,331ff907ebce4ca5a463980ed131de88]
    POC    3 TId: 0 ( I-SLICE, nQP 47 QP 47 )      11976 bits [Y 28.0543 dB    U 38.4835 dB    V 39.0113 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8e8f11283f8bf437f3aa1af4d4cd54d6,2281724d3283a097c64ace9b50d68146,92fcd1cbe52cad9fe6c495bdf36cba5b]
    POC    4 TId: 0 ( I-SLICE, nQP 47 QP 47 )      13160 bits [Y 25.1441 dB    U 38.5267 dB    V 39.3923 dB] [ET     2 ] [L0 ] [L1 ] [MD5:35473b944c5f407d2dfa5a3daf361a2b,378bea8c962c80061fcb2de28cf52b13,63eb898dbe60be7695495815ac911da3]
    POC    5 TId: 0 ( I-SLICE, nQP 47 QP 47 )       7872 bits [Y 29.2276 dB    U 39.7486 dB    V 38.8858 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e079eaf0c99946872ea27660c5ed8687,5466ae88a8be61cfd7c95d6c051db9d5,7a7f2d03544138f1fe2577543b205414]
    POC    6 TId: 0 ( I-SLICE, nQP 47 QP 47 )      28864 bits [Y 22.1011 dB    U 36.2977 dB    V 38.6267 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bbf685647bdd7cfc8139d782a205c20f,19b562d803f30f04ea46668136830b8a,f7091a9587a07996117b2ea06a61dd52]
    POC    7 TId: 0 ( I-SLICE, nQP 47 QP 47 )      16864 bits [Y 27.7117 dB    U 35.6273 dB    V 37.5592 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fee3f2679b06e4d1c85b91c4ed375369,52025d5274e170fcf9ffeac8bf39d30f,41715b1a506600f35dac6b0799681349]
    POC    8 TId: 0 ( I-SLICE, nQP 47 QP 47 )       8160 bits [Y 29.4593 dB    U 37.8309 dB    V 38.4685 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c4c23cd486702477ee6aa35989833699,f2da42a2391e521fff6e64437eb65b99,0d2270af651ffdb023a926cef8d3de35]
    POC    9 TId: 0 ( I-SLICE, nQP 47 QP 47 )      12888 bits [Y 27.4571 dB    U 39.5927 dB    V 40.1589 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e12b47b27888295631dd24f5a2c4279b,d2ec393a0b285e568be8fa3eb0297f85,e8d757b049241fcf3623d152f5db2876]
    POC   10 TId: 0 ( I-SLICE, nQP 47 QP 47 )       7048 bits [Y 27.7607 dB    U 39.9535 dB    V 41.5147 dB] [ET     2 ] [L0 ] [L1 ] [MD5:90ece1048fcdc52f676310266b1c599a,a1be263885d688be50f02472d0469ece,c3506364ee9887826312dc6f6849e42e]
    POC   11 TId: 0 ( I-SLICE, nQP 47 QP 47 )       6088 bits [Y 29.2285 dB    U 37.2096 dB    V 34.9737 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c70ba422c0d80ac29391b3b00f9b3b8f,9912ae6eb7fd3462e77f54e22ac6d766,56af78b436bba0a021a30958d61ae041]
    POC   12 TId: 0 ( I-SLICE, nQP 47 QP 47 )      21624 bits [Y 25.1166 dB    U 34.6535 dB    V 34.7645 dB] [ET     2 ] [L0 ] [L1 ] [MD5:768ff04754c8ea11aea69187a03c2e02,dce5790e8672c98dd77735617ed47bc2,8af8439ef06074c93fcf547342e2acc5]
    POC   13 TId: 0 ( I-SLICE, nQP 47 QP 47 )      17912 bits [Y 23.9208 dB    U 38.9496 dB    V 37.1922 dB] [ET     2 ] [L0 ] [L1 ] [MD5:79eff56d28869ee1c2f67f76f745a8d2,9822954ff5571d2ab17233cf8e99a951,bdbede688d2e62d3c78e8f834360089f]
    POC   14 TId: 0 ( I-SLICE, nQP 47 QP 47 )      10656 bits [Y 28.7127 dB    U 38.8809 dB    V 35.7534 dB] [ET     2 ] [L0 ] [L1 ] [MD5:de3151784f908d462a5cbe61fed61bc4,9e360b9b68cc43e486a655ce3e7d7366,922bbc1a765019c9372462ea8a02e0e4]
    POC   15 TId: 0 ( I-SLICE, nQP 47 QP 47 )      20224 bits [Y 24.4827 dB    U 36.2263 dB    V 37.4309 dB] [ET     2 ] [L0 ] [L1 ] [MD5:05310df81b0d3bfdd750a84aad957676,5e90de47f1ceadad2d9fe9627f8ff2ee,2d5c775d5d0210bbd3a262b905b7bce1]
    POC   16 TId: 0 ( I-SLICE, nQP 47 QP 47 )      16936 bits [Y 24.6635 dB    U 35.9055 dB    V 36.5923 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1592d47ed04b8cb7358bd5211e43ed4a,34cd9b5f787e4995f902f359e396bc67,96d302f593ed496d5e56822ca558e24e]
    POC   17 TId: 0 ( I-SLICE, nQP 47 QP 47 )      14600 bits [Y 27.1190 dB    U 38.5324 dB    V 39.1856 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d797b483ac3ad3307954fe4e2795088f,1c991e8b1e762cf8024d0966ad2fc205,cc85ee58c3e7d054cd4ecfddf65050a9]
    POC   18 TId: 0 ( I-SLICE, nQP 47 QP 47 )      11368 bits [Y 26.9984 dB    U 36.5550 dB    V 36.7957 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bb7afe6e42d4f5678e1e29663d3bd0a5,f817e6f43cefd2d1779dede6915c3317,a60a2546a1d99611ec167579ca0ec201]
    POC   19 TId: 0 ( I-SLICE, nQP 47 QP 47 )      12112 bits [Y 30.2109 dB    U 36.8795 dB    V 36.8748 dB] [ET     2 ] [L0 ] [L1 ] [MD5:eff6683bffe6893dc2c6624ac1cbcae1,b2fbdb3603e91db47accd52c6ff3b20c,3f14a16184f15876174207fd6dc09f44]
    POC   20 TId: 0 ( I-SLICE, nQP 47 QP 47 )      11232 bits [Y 28.4445 dB    U 38.5014 dB    V 40.3253 dB] [ET     2 ] [L0 ] [L1 ] [MD5:62c3b39e72d240ea511dd5f187f60526,e995a5feb8388328c0b11dfb0d9d68b2,9f67f4f2b888c18e1fbb92b5a5d67a83]
    POC   21 TId: 0 ( I-SLICE, nQP 47 QP 47 )      17016 bits [Y 25.6362 dB    U 37.8029 dB    V 38.6757 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6429e68b943a89d7dccb78e29913abb2,74e19c278d0b61719db6d50c56fb1e8f,eafb6062e009fc1cc489df0fc856eec6]
    POC   22 TId: 0 ( I-SLICE, nQP 47 QP 47 )      11296 bits [Y 28.4450 dB    U 37.8004 dB    V 40.1709 dB] [ET     2 ] [L0 ] [L1 ] [MD5:542d132fcbe1b4cfdf027fdcccfd5d1d,e57c9d8f6d66bcd741fbb797b9124687,0b5b0101721ac58de3eb2997612383d3]
    POC   23 TId: 0 ( I-SLICE, nQP 47 QP 47 )      40008 bits [Y 23.2427 dB    U 36.0746 dB    V 34.9907 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d8dd89be89964c99f929c0aa971f6af4,8c9d70db2a9d15d1377cf8dbf6545caf,9757bc2bbf053fddb493c22302484b51]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a     937.9000   26.7286   37.6180   37.6193   27.7281  

# 46
    POC    0 TId: 0 ( I-SLICE, nQP 46 QP 46 )      15784 bits [Y 26.5496 dB    U 38.8771 dB    V 36.6988 dB] [ET     2 ] [L0 ] [L1 ] [MD5:eb8fe741545a6f963b337c2c3f1a2646,f96321d47ebe3c8fc2c4a63ebdbda15d,b2b847e9a543304c32391a3472362f9f]
    POC    1 TId: 0 ( I-SLICE, nQP 46 QP 46 )      40752 bits [Y 23.9109 dB    U 34.7946 dB    V 34.9957 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5fe716947b956be9cae1fb0b0f9d281e,8473cb2b66c5feee7495719dcdfc0a7a,4650882f4e9a86de4d3089a2eb76ac02]
    POC    2 TId: 0 ( I-SLICE, nQP 46 QP 46 )      11904 bits [Y 29.1992 dB    U 40.1222 dB    V 35.2445 dB] [ET     2 ] [L0 ] [L1 ] [MD5:80593e880b3252d6f4f45438cbe170d9,f31e2b96a5182b0e79de6af2ef0a16ff,3179b68a4089eb3686776047752e656f]
    POC    3 TId: 0 ( I-SLICE, nQP 46 QP 46 )      14304 bits [Y 28.6786 dB    U 38.9439 dB    V 39.4956 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dd43da11423533e379f4ecd2442cd43a,d5a532f8025bb787480c9e9bb7985b2c,08e9925698351d4a22ed81eca2bc187e]
    POC    4 TId: 0 ( I-SLICE, nQP 46 QP 46 )      16480 bits [Y 25.6441 dB    U 38.9894 dB    V 39.9569 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dffea528e2e7992182b148e4a1beb0e3,575819870ddebc9712257bf27772f275,5693f6132c96011b00205f504f99eacb]
    POC    5 TId: 0 ( I-SLICE, nQP 46 QP 46 )       8968 bits [Y 29.6650 dB    U 39.7061 dB    V 39.0841 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cf5ea987b42fee60abaf9e79e3caa29e,83430ba171ccfda290410426dc1a71a7,c3af4868da43fb03387280e2e7762be6]
    POC    6 TId: 0 ( I-SLICE, nQP 46 QP 46 )      36512 bits [Y 22.5186 dB    U 36.5520 dB    V 39.0654 dB] [ET     2 ] [L0 ] [L1 ] [MD5:737ee8762432025bb030529314d600c8,df2540d8f8a1de61e54832419b9c6fee,c497176c901942c32134d415ed85b5cc]
    POC    7 TId: 0 ( I-SLICE, nQP 46 QP 46 )      19912 bits [Y 28.2291 dB    U 36.1142 dB    V 38.0979 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3286d1a2556eb77b9d57fb32e2de6a2a,43b5a0cc0da86a363182e13295ee951f,26f9c561fbb201762abcbf7defb40eb8]
    POC    8 TId: 0 ( I-SLICE, nQP 46 QP 46 )       9800 bits [Y 29.9756 dB    U 38.6619 dB    V 39.1864 dB] [ET     2 ] [L0 ] [L1 ] [MD5:426d480cb966b7cb162970d67bbdf623,1e12fa2a24a7b4c453b1b90e14e5a10a,4a67a593a13ba95459d0a93e833996c0]
    POC    9 TId: 0 ( I-SLICE, nQP 46 QP 46 )      15712 bits [Y 27.9569 dB    U 40.1338 dB    V 40.8119 dB] [ET     2 ] [L0 ] [L1 ] [MD5:941f7120d57b16bee08482ea9e0271ca,92d5d6bf8c5b7389999ffd4ef7043661,8db6034707771d29ebdfd0be9f10173c]
    POC   10 TId: 0 ( I-SLICE, nQP 46 QP 46 )       8520 bits [Y 28.0623 dB    U 40.7472 dB    V 41.8814 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2212dbd3010dad06335182d865795d5c,6d2e6caf2e25987e74e643b0c5fcd187,fe102d1641d7a3e9f6d179e5eb470d28]
    POC   11 TId: 0 ( I-SLICE, nQP 46 QP 46 )       7440 bits [Y 29.6229 dB    U 37.7044 dB    V 35.7106 dB] [ET     2 ] [L0 ] [L1 ] [MD5:120e87c3bcff01d8752dd0ea1846131f,cc9f8daed4c2f687c0c88ad221217eeb,e721a00d7f81904d29bc1ac4cc9b1401]
    POC   12 TId: 0 ( I-SLICE, nQP 46 QP 46 )      26368 bits [Y 25.6501 dB    U 35.3198 dB    V 35.3910 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cb3f39d205b57c49dd791336bfd7abaa,42eebfe4a48d5f5c4a08bbacb0f8f264,390b0d9b5708cf98c1408a89fe974a62]
    POC   13 TId: 0 ( I-SLICE, nQP 46 QP 46 )      22280 bits [Y 24.3266 dB    U 39.7468 dB    V 37.9463 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5b4a2c42d16a0f4c1f27d4cbd2fe91f5,a0b4d971dd3ec88f375a9d7189fd3f0f,e989e641464bf9bdb8746845b28bfc34]
    POC   14 TId: 0 ( I-SLICE, nQP 46 QP 46 )      12272 bits [Y 29.1628 dB    U 39.3464 dB    V 36.2945 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a3de86be302b5efed10797036e94f87a,750d7dfb9251e1d5139a77bce21858e4,01993a9496222624af65cd8abed9305c]
    POC   15 TId: 0 ( I-SLICE, nQP 46 QP 46 )      25040 bits [Y 24.9341 dB    U 36.4988 dB    V 37.9600 dB] [ET     2 ] [L0 ] [L1 ] [MD5:679b1c29efb868a834724021846fd0be,90edbaca2ced908be796ba3a822c66f0,4429b4ec12355050ba25b1e9c6b7a90f]
    POC   16 TId: 0 ( I-SLICE, nQP 46 QP 46 )      20856 bits [Y 25.0123 dB    U 36.2044 dB    V 36.8565 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2423bdf89572af1087370e44d396f770,910433d1cb5acd1b087886c7f15e2cdc,9c62d17a9e5b501ee3560a4f468ebd98]
    POC   17 TId: 0 ( I-SLICE, nQP 46 QP 46 )      16648 bits [Y 27.4895 dB    U 39.2124 dB    V 39.6532 dB] [ET     2 ] [L0 ] [L1 ] [MD5:016007c2fdca744cf25a71a7b6986985,84d4f7fc90a3c19ed9185c1198eb03cb,2fb317a9d77b8a17853a138dbe4be69e]
    POC   18 TId: 0 ( I-SLICE, nQP 46 QP 46 )      13504 bits [Y 27.3301 dB    U 36.9547 dB    V 37.2109 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8f4647a8a679e50eb62f3727b9087d33,70560d0efadda90c57e9d60bc5fd29b7,df41dc66e038d4c9014ca55a57473213]
    POC   19 TId: 0 ( I-SLICE, nQP 46 QP 46 )      13864 bits [Y 30.7022 dB    U 37.8103 dB    V 37.3483 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3f579b7680a3a81b5a04271fe3d7c305,be464941a01811026b2dea3028f7f182,32701f8281c5ed7348014df54efa7f4a]
    POC   20 TId: 0 ( I-SLICE, nQP 46 QP 46 )      13264 bits [Y 28.9378 dB    U 39.1085 dB    V 40.4624 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f71ef155854b4d3186f67d774d4dfb66,5c67f8551f0d0f070a6df379bb4b6541,29425216fe281a0b6751f0e49fdf5744]
    POC   21 TId: 0 ( I-SLICE, nQP 46 QP 46 )      20408 bits [Y 26.0470 dB    U 37.9532 dB    V 39.0033 dB] [ET     2 ] [L0 ] [L1 ] [MD5:840d2cba40da0916ad1a2d6532431d04,08ec3511f60578b935b8c99f25404057,959a92a8434ad0acbb4c6ef91ad717ff]
    POC   22 TId: 0 ( I-SLICE, nQP 46 QP 46 )      12752 bits [Y 28.9360 dB    U 37.9017 dB    V 40.6722 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f2584b9b07d80296ef16f0dfb4d3044b,0c66f9c93b6bb63641ff85ef1415b51f,f16f6f0e580488c856011e0e4adf1cf0]
    POC   23 TId: 0 ( I-SLICE, nQP 46 QP 46 )      46464 bits [Y 23.7797 dB    U 36.0500 dB    V 35.4771 dB] [ET     2 ] [L0 ] [L1 ] [MD5:beb7b546ff034b702107d93218337e6f,5fb6330d56284cb4ee17ee5f99c24857,b55b829f3184b04e84daa957decba280]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    1124.5200   27.1800   38.0606   38.1044   28.1801  

# 45
    POC    0 TId: 0 ( I-SLICE, nQP 45 QP 45 )      19304 bits [Y 27.0290 dB    U 39.4003 dB    V 37.0250 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ab8e712212945d856aae0e11035507ac,eed13e3af7220105f63aec2920740ae6,4df55e2bde62cce523147c6caf444fa8]
    POC    1 TId: 0 ( I-SLICE, nQP 45 QP 45 )      48000 bits [Y 24.4050 dB    U 35.1369 dB    V 35.4354 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ce41e07215e846e79f4063babb5668e8,de4fc2df8fabb40b6013411e0acf347a,d56f83aebdde5d434010656ae99a3de5]
    POC    2 TId: 0 ( I-SLICE, nQP 45 QP 45 )      13120 bits [Y 29.4759 dB    U 40.6917 dB    V 35.7988 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cc0dc70acade18bbbe2fe5abb2cff39b,27142dffd9c4a73be9f162f7ef83795a,df00b708ba2fca57af8613f098bba06b]
    POC    3 TId: 0 ( I-SLICE, nQP 45 QP 45 )      16976 bits [Y 29.2383 dB    U 39.1920 dB    V 39.9346 dB] [ET     2 ] [L0 ] [L1 ] [MD5:67d8edebec22d0775e85352fce069b82,7f70e692b5ac3402bf72c87526ec429e,d0560ccc9fdd08138ffd22b17dfb6f7f]
    POC    4 TId: 0 ( I-SLICE, nQP 45 QP 45 )      20360 bits [Y 26.0572 dB    U 39.2882 dB    V 40.0377 dB] [ET     2 ] [L0 ] [L1 ] [MD5:12f3f6e4d85c35e34b06d8747401b78d,df8a9dd9c2a4c845f752d779a8fcb54f,e46ecf27e476bbfd329e2db6eff46b78]
    POC    5 TId: 0 ( I-SLICE, nQP 45 QP 45 )      10240 bits [Y 30.0686 dB    U 40.1014 dB    V 39.8109 dB] [ET     2 ] [L0 ] [L1 ] [MD5:83d6d63bff126930cb00d01f1accc1cf,382dc7eab0586553fa68c946394c04cf,197acea1d7e2f6d947fe5618565f2cbd]
    POC    6 TId: 0 ( I-SLICE, nQP 45 QP 45 )      46136 bits [Y 22.9703 dB    U 36.7399 dB    V 39.3337 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d7c7e5a126d41a1c7d4617e63367acb7,e59685a06e3f51213435281132f948e7,88ef3dab199580d5ac7108c7a86ddc89]
    POC    7 TId: 0 ( I-SLICE, nQP 45 QP 45 )      23000 bits [Y 28.7984 dB    U 36.4890 dB    V 38.5871 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7422ee735c0d49563925f4ab86ad770e,c84b89ef092e523f5038d8355b33c2b5,fab3b910bbca4412265b7dc8de57bd58]
    POC    8 TId: 0 ( I-SLICE, nQP 45 QP 45 )      10936 bits [Y 30.4476 dB    U 39.1012 dB    V 39.7377 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9441ab25ae3a7e8b8f35258945b2651d,aa435085121ddb3cbd90d4a56b710024,b7f7499767bd2eb4e72948f1c03ed1dd]
    POC    9 TId: 0 ( I-SLICE, nQP 45 QP 45 )      18600 bits [Y 28.4259 dB    U 40.7404 dB    V 41.1099 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b264ae5fa8431d5a911dc6508d80fe28,4fbe38e466e7d190dc97871a23d905e0,adb8df9f5519893266ffaa3758821ed1]
    POC   10 TId: 0 ( I-SLICE, nQP 45 QP 45 )      10216 bits [Y 28.3953 dB    U 40.7262 dB    V 41.8445 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5bc10a031a5f7def4a1e2648539ed563,b3dc8f93e7eb2e3bd2132d460a1bd6ad,de8bb09e691c672b58ad9bda953b6498]
    POC   11 TId: 0 ( I-SLICE, nQP 45 QP 45 )       8728 bits [Y 30.1641 dB    U 38.6605 dB    V 36.1000 dB] [ET     2 ] [L0 ] [L1 ] [MD5:67ddc6692365b808e80c44219237d605,6adffbffa9ac628ab700a3cc7ef8cf6e,f1657d0996c531763dad416d9b705971]
    POC   12 TId: 0 ( I-SLICE, nQP 45 QP 45 )      31104 bits [Y 26.0652 dB    U 35.6968 dB    V 35.9995 dB] [ET     2 ] [L0 ] [L1 ] [MD5:163ebdbbb579a2a523943a6b9ad2d82a,939e0ae82d68b498294667f4446a14ec,e4c05b3a61920fa87fa77bbee6e694d8]
    POC   13 TId: 0 ( I-SLICE, nQP 45 QP 45 )      27072 bits [Y 24.7152 dB    U 39.6947 dB    V 37.9406 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0796c186c877284c145400e7934c9af4,624e0c1eb7674791c3abe9db7c587397,970bdb628dd2f80bdb1097eb22bd6d48]
    POC   14 TId: 0 ( I-SLICE, nQP 45 QP 45 )      13792 bits [Y 29.5833 dB    U 39.4394 dB    V 36.6170 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0761e68104044be5053478dfa30f32e7,ea67c05c84e8b575d3c6b099907c7f4f,69f69589e22f31ca642569e6193393e5]
    POC   15 TId: 0 ( I-SLICE, nQP 45 QP 45 )      31080 bits [Y 25.4128 dB    U 36.6771 dB    V 38.0034 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9d94969710ef31ae95bf395fee0a2c38,a3f0dd08c25fcdf8cfc5a29813399771,8ac8284d3a21b6c01015384972e5106d]
    POC   16 TId: 0 ( I-SLICE, nQP 45 QP 45 )      26432 bits [Y 25.4610 dB    U 36.5501 dB    V 37.1872 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b4078f5e1d5945acea41b0aa6e826b62,508558c71a00ba9f596f89db7d47b460,8dd7bf0e3bcb5933c7b0029e96b72095]
    POC   17 TId: 0 ( I-SLICE, nQP 45 QP 45 )      19440 bits [Y 27.8945 dB    U 39.7978 dB    V 40.1502 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8ec699d4781749dd683d7e5c6458e8c0,0ce75c28bc0e8e588b2a38b08e2a5ac9,41292ad0d1ed3bc089cf082cba5c6d3b]
    POC   18 TId: 0 ( I-SLICE, nQP 45 QP 45 )      15904 bits [Y 27.6858 dB    U 37.3073 dB    V 37.5117 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8e4a9c214212e9cdc07f7d4264d7a7f9,f8724bd5e1f29fca633e32d5fd384ab7,2426126a1973ae44947cf024c033bb45]
    POC   19 TId: 0 ( I-SLICE, nQP 45 QP 45 )      15792 bits [Y 31.2040 dB    U 38.2681 dB    V 37.9475 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9dae8a5505ad09f659930b0382b93e12,e1453c2ec8907d1f8e47526aed811101,92d745313c15c105e35ef4ecadd1fe1c]
    POC   20 TId: 0 ( I-SLICE, nQP 45 QP 45 )      15824 bits [Y 29.5356 dB    U 39.8485 dB    V 41.0487 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7a19863c80d32321f4d17131ac73219e,ad34b4e6e86f91343adba27b9a6d8cee,71562075741a683b02d801d160314720]
    POC   21 TId: 0 ( I-SLICE, nQP 45 QP 45 )      24568 bits [Y 26.4859 dB    U 38.5715 dB    V 39.6311 dB] [ET     2 ] [L0 ] [L1 ] [MD5:45dc52d6c50d43b544872431beead96d,37d01ed58743feed3efa820b94c19327,5cc4ded43882bf93bafb81b6e7e3b746]
    POC   22 TId: 0 ( I-SLICE, nQP 45 QP 45 )      14648 bits [Y 29.5052 dB    U 38.5074 dB    V 41.0022 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e223581670450e29bfb8dd74a593f82d,25f473f4af449f4fb6cb5a305375652c,adc1b248cb40aa20e0ce612428032d31]
    POC   23 TId: 0 ( I-SLICE, nQP 45 QP 45 )      55360 bits [Y 24.4041 dB    U 36.6522 dB    V 35.6797 dB] [ET     2 ] [L0 ] [L1 ] [MD5:040f4100c8685b3b9a3ab8c84734173e,ea48d07d9b6a2b46cecdc6d9bca14a6a,8977bf304ebde14edf55cf948a2b390f]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    1341.5800   27.6428   38.4699   38.4781   28.6411  

# 44
    POC    0 TId: 0 ( I-SLICE, nQP 44 QP 44 )      23064 bits [Y 27.4802 dB    U 39.7578 dB    V 37.7073 dB] [ET     2 ] [L0 ] [L1 ] [MD5:21cd06cd432d3b20426ebab09a227076,0616a2e82035cb02f7d70aa1d532b4b1,7931939d94db75c8e18f1dd44508f181]
    POC    1 TId: 0 ( I-SLICE, nQP 44 QP 44 )      57576 bits [Y 24.9918 dB    U 35.5367 dB    V 35.7418 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ac914fcd6cf435ba6b8abc49237e60cb,cdb760a9bc87fbd22269bd832fb92b71,53263c9927dc007acaa4d619b599ae3e]
    POC    2 TId: 0 ( I-SLICE, nQP 44 QP 44 )      15336 bits [Y 29.8234 dB    U 41.1294 dB    V 36.3582 dB] [ET     2 ] [L0 ] [L1 ] [MD5:89e4541060cf3ac4046908a0cd78a0f9,8f3016db9e5c5b0b7d3f6557da0ef4b5,acf6c14e03d30282b1040177675215f1]
    POC    3 TId: 0 ( I-SLICE, nQP 44 QP 44 )      19600 bits [Y 29.6665 dB    U 39.6332 dB    V 40.2146 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e56aa4c707f6b3404447c41551249661,17390982e7d7c06383127deeb8e2265f,70871913091747c5df55001a821a63f1]
    POC    4 TId: 0 ( I-SLICE, nQP 44 QP 44 )      24960 bits [Y 26.5161 dB    U 39.4836 dB    V 40.4841 dB] [ET     2 ] [L0 ] [L1 ] [MD5:09828488fafe3c1300be59686ee7fb63,dd32a8ba40d811271fe40e7ee114008a,64f9f1dc5dc6d184e862ef81361af330]
    POC    5 TId: 0 ( I-SLICE, nQP 44 QP 44 )      11808 bits [Y 30.5955 dB    U 40.7229 dB    V 40.2829 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5e2012ca86fa03b81df6f3417945e309,919c84023569d4a66fad4bb2a4649c92,883a58be5456093a92c38502e1914416]
    POC    6 TId: 0 ( I-SLICE, nQP 44 QP 44 )      57104 bits [Y 23.4299 dB    U 37.0434 dB    V 39.5916 dB] [ET     2 ] [L0 ] [L1 ] [MD5:52b3fc472799d34689789dc40a907117,e0ba446d0efe7bfcc6dc602d55894d99,78fcee89aca21a6e8c51e14a8d48db88]
    POC    7 TId: 0 ( I-SLICE, nQP 44 QP 44 )      26208 bits [Y 29.2685 dB    U 37.0622 dB    V 39.3871 dB] [ET     2 ] [L0 ] [L1 ] [MD5:94bdb5ddff730c97dee06f9f2d9f8908,c00a9f4232dc07dde1c803b42977e38e,531851448414c0cd3721df85574d1000]
    POC    8 TId: 0 ( I-SLICE, nQP 44 QP 44 )      12656 bits [Y 30.8384 dB    U 39.6036 dB    V 40.3377 dB] [ET     2 ] [L0 ] [L1 ] [MD5:48d63ce8f33ef34e7141acbff00286c0,74da83346de563fe4c95967cb0886ac3,4c2e7f644064468398a6218b4f9a555c]
    POC    9 TId: 0 ( I-SLICE, nQP 44 QP 44 )      21712 bits [Y 28.8569 dB    U 41.0686 dB    V 41.5172 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9afb5d62efa39c39379956967f604780,24909f8b28bec3e48c1013f1e59a7e86,1324b9af663f82e2670f1b7fa7d13078]
    POC   10 TId: 0 ( I-SLICE, nQP 44 QP 44 )      12952 bits [Y 28.7696 dB    U 41.0453 dB    V 42.6124 dB] [ET     2 ] [L0 ] [L1 ] [MD5:80bd54181493d88102ef4277ecd79540,e43450e33f34041e77eb0182258c2a5c,e43bef7e4c19e86a2519fce5a018a20c]
    POC   11 TId: 0 ( I-SLICE, nQP 44 QP 44 )      10072 bits [Y 30.4507 dB    U 39.2117 dB    V 36.4407 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2c34e7ff9332225fae38dacfbcd93601,039c981cb72cc9e0a12d14e9358f68d9,033aa6c02d11532836fecf826b4fa3a2]
    POC   12 TId: 0 ( I-SLICE, nQP 44 QP 44 )      36936 bits [Y 26.4837 dB    U 36.0978 dB    V 36.4966 dB] [ET     2 ] [L0 ] [L1 ] [MD5:003c8ff4578beb55596e0d041a3906a4,ca1abee9dc8293f8169659c62c425912,487859e26c293b9efbbc707dcec6d39a]
    POC   13 TId: 0 ( I-SLICE, nQP 44 QP 44 )      33448 bits [Y 25.1376 dB    U 40.1210 dB    V 39.0204 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9dcde75c71e5b21405257f1209e2d2b1,00b79103f0cae9e58ccc62aa052f36f6,371c71dd6d727dd9d5d3f50491ed30d1]
    POC   14 TId: 0 ( I-SLICE, nQP 44 QP 44 )      15976 bits [Y 30.0302 dB    U 40.0537 dB    V 37.1132 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0e2483b35fdaec4327a75abb4fe8b2ed,7d3cf1cdb3b674c63d2f767bd9ec992f,f34a77307e9d85d64def76bebbc382f1]
    POC   15 TId: 0 ( I-SLICE, nQP 44 QP 44 )      38056 bits [Y 25.9244 dB    U 36.9905 dB    V 38.2675 dB] [ET     2 ] [L0 ] [L1 ] [MD5:091e606d3eeefdf4080b28deac61aa58,c235ffca993d2049b8c52969212aabc6,c7f2526af4a9db41dcb76815d4fada8a]
    POC   16 TId: 0 ( I-SLICE, nQP 44 QP 44 )      32728 bits [Y 25.9012 dB    U 36.7867 dB    V 37.5021 dB] [ET     2 ] [L0 ] [L1 ] [MD5:556c4323c40531e896141a85f7f3dbb3,65132d453d2e9ea890678a5a464d83cc,57cb675cba720f35f86cbb2f76ced2fd]
    POC   17 TId: 0 ( I-SLICE, nQP 44 QP 44 )      22512 bits [Y 28.3253 dB    U 40.1615 dB    V 40.5259 dB] [ET     2 ] [L0 ] [L1 ] [MD5:07be740689c0fc1f12658dab9d61e9b8,691c089a1bcbc85e1c7c4035cfe6ef6c,badbb161740bef57029c5ac11abbaaa6]
    POC   18 TId: 0 ( I-SLICE, nQP 44 QP 44 )      19336 bits [Y 28.0115 dB    U 37.7395 dB    V 38.1603 dB] [ET     2 ] [L0 ] [L1 ] [MD5:29a6d4bb7929e642d5b0447d13525713,707be51566610e5bb9c793a4cf5e50c5,049941048ea04f565fb4fd1e77775055]
    POC   19 TId: 0 ( I-SLICE, nQP 44 QP 44 )      17968 bits [Y 31.8058 dB    U 38.7034 dB    V 38.5485 dB] [ET     2 ] [L0 ] [L1 ] [MD5:228d6ad4a1baca39a6ac97fa8db2baf5,f00a9364d6a7aefbb6dfd9340fa317d8,fd7901203712b2a9dc184fc1302759dc]
    POC   20 TId: 0 ( I-SLICE, nQP 44 QP 44 )      17960 bits [Y 30.0033 dB    U 40.1187 dB    V 41.7002 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6aedcd0f75539cf7baff38d90a09bb2b,f254f55dd6167112b283af5d2966bbee,6f3e576c418b4d8deccbfe4481d86518]
    POC   21 TId: 0 ( I-SLICE, nQP 44 QP 44 )      29888 bits [Y 26.9926 dB    U 38.8304 dB    V 39.9293 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2dec40f604b7e454d53f56ddc937e68d,b6b4f37058c70ee0ae7d5b63eb64e5ab,e92d64e923b158155d666de00cd91c91]
    POC   22 TId: 0 ( I-SLICE, nQP 44 QP 44 )      17240 bits [Y 29.9966 dB    U 39.2451 dB    V 41.4286 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8fc36a032757401b1e8df755373b63f8,0da55a98e9993b5a97e0f8bcc8b2c040,4197a86638ba3331719ffdeb99930fb3]
    POC   23 TId: 0 ( I-SLICE, nQP 44 QP 44 )      64312 bits [Y 24.9666 dB    U 36.8421 dB    V 36.1430 dB] [ET     2 ] [L0 ] [L1 ] [MD5:13ccfbbd35a72d44f860cd628b04e274,78bdd705407c7d1acd9913aea0168b9c,ad3be2f4848cb7348734665ba761fc87]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    1598.5200   28.0944   38.8745   38.9796   29.1031  

# 43    
    POC    0 TId: 0 ( I-SLICE, nQP 43 QP 43 )      28472 bits [Y 28.0093 dB    U 39.9362 dB    V 37.9537 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6f45987fd972c86e7c99b9c5c2f72394,7b33f496dc9fb75c246a2793af5d53ab,7ebf809d57c4f31464f123dc19babd4d]
    POC    1 TId: 0 ( I-SLICE, nQP 43 QP 43 )      68688 bits [Y 25.5608 dB    U 36.1243 dB    V 36.4624 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f67462688ac1991510997a8b39922afd,e903ecc23d339bd745353f203d98dfcc,c3a49b3c813f686762161d6f138b73ab]
    POC    2 TId: 0 ( I-SLICE, nQP 43 QP 43 )      17808 bits [Y 30.1998 dB    U 41.5969 dB    V 36.7835 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b8360fd187e1868c5bebb8608846804b,31f1db7d7c92d1381a6962bd59408c35,6b45e3eef21825105e224d3d7997908f]
    POC    3 TId: 0 ( I-SLICE, nQP 43 QP 43 )      22544 bits [Y 30.1666 dB    U 40.1361 dB    V 40.7564 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ac3e37b8c7ff781f133ddb21edbaa5f0,f67a11efafd88866e305177a6d68d2d2,e198e4d4226983ce377fbcb8540efda8]
    POC    4 TId: 0 ( I-SLICE, nQP 43 QP 43 )      30744 bits [Y 26.9378 dB    U 40.1780 dB    V 40.9854 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ccf9a86132a753b04da8c495fb689dac,2919cbdcb45cbd61aeb431dc87798371,55ad497febd6967d673a4f761ca7f296]
    POC    5 TId: 0 ( I-SLICE, nQP 43 QP 43 )      14136 bits [Y 31.0860 dB    U 41.2251 dB    V 40.9157 dB] [ET     2 ] [L0 ] [L1 ] [MD5:069c6da3f4d73ab9e862e3dd9ffaa55d,33422641782089f99fcd817cea77412b,1ecad398fafea07bbea656af3ab8c9aa]
    POC    6 TId: 0 ( I-SLICE, nQP 43 QP 43 )      71664 bits [Y 23.9486 dB    U 37.2908 dB    V 39.7932 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6d8f6443ebbc253a90e10a6e5ffeaa16,d5c1dc6f7c23b275dee0984304b6755f,3c4542add199a343fd9b6022dfd9404d]
    POC    7 TId: 0 ( I-SLICE, nQP 43 QP 43 )      30832 bits [Y 29.9375 dB    U 37.7341 dB    V 39.4859 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0088f99e8d5dda265c6566c90658c6bb,3fc6c2d2997985bd33af83d01e369816,63502ab4ccbcfa18d87b27276da1fd82]
    POC    8 TId: 0 ( I-SLICE, nQP 43 QP 43 )      14928 bits [Y 31.1796 dB    U 40.0592 dB    V 40.8659 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3f43554b6ffa49c63ccd07b532138c3d,161b70d1a2fff24d7acedda8948251e6,da10c82564d8120ba61e7ca48482afc5]
    POC    9 TId: 0 ( I-SLICE, nQP 43 QP 43 )      25424 bits [Y 29.4665 dB    U 41.5410 dB    V 41.8874 dB] [ET     2 ] [L0 ] [L1 ] [MD5:eedd916a1598375a4ff9beca56644de6,4f7eae9d1bf6c93018b770fb29e29b30,984a0995390b9fa7088c6304ebba4b89]
    POC   10 TId: 0 ( I-SLICE, nQP 43 QP 43 )      15920 bits [Y 29.1802 dB    U 41.7348 dB    V 43.0701 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0541005de1ef5334ecd8cebb6c1c0862,082f941cfcdc6e8206f6315505b7cc95,39abc133073819474b337f473b438504]
    POC   11 TId: 0 ( I-SLICE, nQP 43 QP 43 )      12136 bits [Y 30.7267 dB    U 40.5664 dB    V 36.9933 dB] [ET     2 ] [L0 ] [L1 ] [MD5:336a9e74fc2d1b334a6773aa3a06b98e,a4f595f872237fc14f35b143b4961a2e,c7af0573583efa9079eaedbbcefa05c8]
    POC   12 TId: 0 ( I-SLICE, nQP 43 QP 43 )      43728 bits [Y 26.9665 dB    U 36.8041 dB    V 37.0601 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3bd59ca3a2855f2109b5f21a8fdb6a25,a188642ffc39330ff5d4fcabac625410,57a3f3c55430faf7d86634c6fbecf06d]
    POC   13 TId: 0 ( I-SLICE, nQP 43 QP 43 )      41008 bits [Y 25.5849 dB    U 40.7290 dB    V 39.3824 dB] [ET     2 ] [L0 ] [L1 ] [MD5:85e96990d6ad40cce615fb568b87070e,851b14eab41673e0b5da916bae560a00,7fa0c56e47a9d6db8afe2fb53586ee3d]
    POC   14 TId: 0 ( I-SLICE, nQP 43 QP 43 )      18080 bits [Y 30.3284 dB    U 40.6906 dB    V 37.5267 dB] [ET     2 ] [L0 ] [L1 ] [MD5:71707fb0bcb2d542c2240145b36e0388,3e7f9c9554fcba6a840d83b5fd630e9e,b72112063f3411dcbafa0751be8ece93]
    POC   15 TId: 0 ( I-SLICE, nQP 43 QP 43 )      46560 bits [Y 26.4467 dB    U 37.4127 dB    V 38.7134 dB] [ET     2 ] [L0 ] [L1 ] [MD5:68afc904a2559be36a715d3ca4473c46,ec2f030df90149586b5a8cc9fc607d70,5c9e81b08334d2fde98ceabfa6e028dc]
    POC   16 TId: 0 ( I-SLICE, nQP 43 QP 43 )      41064 bits [Y 26.4085 dB    U 37.3114 dB    V 37.7594 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e7de7675cc4299ab85a5b38e8eb604f8,e2f63fe2c3a57971548d614eed6e38c6,fcb4cac906b838685b36e01547252316]
    POC   17 TId: 0 ( I-SLICE, nQP 43 QP 43 )      26232 bits [Y 28.7670 dB    U 40.4495 dB    V 40.8647 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f0ca90d44810b8b436582f18bf7d01af,9fa537d70fe5835779fbae1c4cd15df7,3f0725ceb1ff732a8706ed024b0fefd2]
    POC   18 TId: 0 ( I-SLICE, nQP 43 QP 43 )      23280 bits [Y 28.4248 dB    U 38.1121 dB    V 38.4664 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b6f734410598b6123738a6ef07bb5a73,9248d74d83e095c438d5549b330e5cb2,d6956dce641f3d0aa2b2a52e7f293096]
    POC   19 TId: 0 ( I-SLICE, nQP 43 QP 43 )      20688 bits [Y 32.3751 dB    U 39.3719 dB    V 39.0603 dB] [ET     2 ] [L0 ] [L1 ] [MD5:147438ebab9e5a470be5cf3358db3860,a1e80f2f9da2e0ace7e53f0cae654df7,14f823db4b574eec8b7a0b70a4b33ae5]
    POC   20 TId: 0 ( I-SLICE, nQP 43 QP 43 )      21096 bits [Y 30.5732 dB    U 40.6066 dB    V 42.2025 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a8d0f522e9cb7e4d629be0c3b7a53c13,b4421c320019a2cc1dfa53ca7fe78266,dd2cf7979ca10b6dc700121d63232f2d]
    POC   21 TId: 0 ( I-SLICE, nQP 43 QP 43 )      36184 bits [Y 27.5015 dB    U 39.0182 dB    V 40.4901 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b2d0517b6febae7c3419f80c0cf3f74d,610bdc8d31d2552323ba62755aa89875,0bc4c15dd308bef3c37bc9fbff97fb99]
    POC   22 TId: 0 ( I-SLICE, nQP 43 QP 43 )      19856 bits [Y 30.4444 dB    U 39.8562 dB    V 41.8482 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3f9917a3160d14d2cb2e3ceee6f2d610,cbf2ae5ee449e3fa2720458ba0001b0f,edf9ddcbb1f5fdade99f0a8707ef1f91]
    POC   23 TId: 0 ( I-SLICE, nQP 43 QP 43 )      74480 bits [Y 25.5542 dB    U 37.3993 dB    V 36.7567 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e13f72ef159aa8bf78c727cf33904dd7,8a7f309d90f598a6633a8cf212e17e6f,7a3af9b7f4a1b795363de2242a828480]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    1913.8800   28.5739   39.4119   39.4202   29.5965  

# 42
    POC    0 TId: 0 ( I-SLICE, nQP 42 QP 42 )      33032 bits [Y 28.4538 dB    U 40.0625 dB    V 37.9552 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d2cbdefe8086643ad39d090577a70d29,e3543914ea83f67e6677c476066dcbfe,3aae8b902d6a658ab1d5ed74f2171b0b]
    POC    1 TId: 0 ( I-SLICE, nQP 42 QP 42 )      79600 bits [Y 26.1661 dB    U 36.1693 dB    V 36.3587 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5142b90696b7e4d1fbbf5b870a4b58da,adeae9057004c05c59df558fd93ad10e,5d18d9dbdb47bfb2c8ae8f1a5d16b49b]
    POC    2 TId: 0 ( I-SLICE, nQP 42 QP 42 )      20520 bits [Y 30.6126 dB    U 41.5918 dB    V 36.6914 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fff806872bb55d8dfa82dc9c76c8f2af,b02beaed65a8c94b913336a3442a8aa8,75c43235c02a51ed1aa87001bfdcce5b]
    POC    3 TId: 0 ( I-SLICE, nQP 42 QP 42 )      26256 bits [Y 30.7607 dB    U 40.0670 dB    V 40.6354 dB] [ET     2 ] [L0 ] [L1 ] [MD5:596fb5b37d61767dcdd96bf5b31fd329,dff877daffe88fa49888170a8cdbbc6e,36854ffb28cf2bdcfeabb8441ba422dc]
    POC    4 TId: 0 ( I-SLICE, nQP 42 QP 42 )      38232 bits [Y 27.4714 dB    U 40.1134 dB    V 40.6770 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fda372b911a6ad09277fd36e69573243,666b5e6eb19f438a7b14c1e3a966b4b6,e3254d9663d7ee2d2738b6cfb993f235]
    POC    5 TId: 0 ( I-SLICE, nQP 42 QP 42 )      15888 bits [Y 31.4092 dB    U 41.3302 dB    V 40.5127 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4463139daca82c75111a0ca6d823427f,5f16c8a77423a653c4f50ab1413e48e7,c921e1a269454b749e7e6de0d267796f]
    POC    6 TId: 0 ( I-SLICE, nQP 42 QP 42 )      87288 bits [Y 24.4873 dB    U 37.2604 dB    V 39.5749 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4e48f3f36998e15891345cdf5ed76be0,be5c898822a6a54ba52ee61dee5a5e2c,751c1b28b153748f7e9d65a78fb8e34e]
    POC    7 TId: 0 ( I-SLICE, nQP 42 QP 42 )      34688 bits [Y 30.5279 dB    U 37.6479 dB    V 39.6740 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fcac0e5cf66fa6787a19dd833ed9eef6,8c15a10561eea3049ac89fdc32a94727,76bc7c37fd329d970142cc75e590c42b]
    POC    8 TId: 0 ( I-SLICE, nQP 42 QP 42 )      16568 bits [Y 31.5906 dB    U 39.8725 dB    V 40.9878 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c9c1a882b0bf3aed54e2551dc6451e2b,b4a2d70c3f1ae0bd0e5c7dbcaa11bdd2,02b11b6c05abe380e97cc228b92cffb4]
    POC    9 TId: 0 ( I-SLICE, nQP 42 QP 42 )      29696 bits [Y 30.0151 dB    U 41.4354 dB    V 41.6448 dB] [ET     2 ] [L0 ] [L1 ] [MD5:277b91c2125f1ed5f2acb4b45e16cae7,4e187e348850f07ed6b585cf7edc9076,3b42122ff2a81251d669d2266f45c0ed]
    POC   10 TId: 0 ( I-SLICE, nQP 42 QP 42 )      19256 bits [Y 29.5640 dB    U 41.4846 dB    V 43.1972 dB] [ET     2 ] [L0 ] [L1 ] [MD5:04c291d6fa351094353402c280576c16,989610e095b1308940e32b6df0e1334d,68c16d6a2854a395a79b774bc9d939d7]
    POC   11 TId: 0 ( I-SLICE, nQP 42 QP 42 )      14056 bits [Y 31.1693 dB    U 39.8324 dB    V 36.9944 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bba66cdd92eb5a61fa2a4274ebc19b0a,5fd5aef0540389507936105d8479e5ea,5f1ef267dd957907a979fffb3c3d9a39]
    POC   12 TId: 0 ( I-SLICE, nQP 42 QP 42 )      50064 bits [Y 27.4535 dB    U 36.7070 dB    V 36.9552 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5d509c664178553f50ecf4b7df4226e3,73f53d86aaf0b9581815beb6876fb6f5,5057da84ea0b332d709946a5200ade64]
    POC   13 TId: 0 ( I-SLICE, nQP 42 QP 42 )      50328 bits [Y 26.0614 dB    U 41.0149 dB    V 39.4650 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b15051561c44d59ed1e51a93b4088031,71abeb425e87422bcf5500904cc50884,33a4a97517e666da8ab92e038594ea5f]
    POC   14 TId: 0 ( I-SLICE, nQP 42 QP 42 )      20528 bits [Y 30.8472 dB    U 40.5532 dB    V 37.5140 dB] [ET     2 ] [L0 ] [L1 ] [MD5:40b091303ea859b7a0b4c56e1ccdafe1,e422bfa02c445a8cc27f38e005b553c6,c3b3fd97545ab1e4e70cf84ac7f6ced7]
    POC   15 TId: 0 ( I-SLICE, nQP 42 QP 42 )      56192 bits [Y 27.0161 dB    U 37.3380 dB    V 38.6706 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6d6303475dd3508b41da25b71bf3498e,438baf6c64338d5ce60aa784a6283c73,776a57afc67bf8026b3587fca0d23a32]
    POC   16 TId: 0 ( I-SLICE, nQP 42 QP 42 )      49456 bits [Y 26.9017 dB    U 37.2447 dB    V 37.8240 dB] [ET     2 ] [L0 ] [L1 ] [MD5:455cd166bf09cb2b9d8f0baa833b7f27,176202bdfe8641ff8456659a6cc1df53,7ee6fcdb8c5d95b27bb06c04cb5325b7]
    POC   17 TId: 0 ( I-SLICE, nQP 42 QP 42 )      29880 bits [Y 29.1817 dB    U 40.3270 dB    V 41.1173 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8d7d03d00546b1243ea3afe0feb011be,51c83b03d918db9d2a5a0fe4a5cb488b,995247e7d986d68ba66d6a475cae12e8]
    POC   18 TId: 0 ( I-SLICE, nQP 42 QP 42 )      27296 bits [Y 28.7762 dB    U 38.0089 dB    V 38.5582 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f1893a00eef644a02d06de311f1d2829,80e27bc94bd784c9a534530e829289b9,1ac2efe0d080e712f1617ae4024af7ab]
    POC   19 TId: 0 ( I-SLICE, nQP 42 QP 42 )      22744 bits [Y 32.9182 dB    U 39.2676 dB    V 39.1359 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9e43c1bdf700b76e6ea699935daa0ccb,2b25fe9fa8d21b16a4e5520e1fc9c1b0,413c80ab94724977372867eca0e903e3]
    POC   20 TId: 0 ( I-SLICE, nQP 42 QP 42 )      23944 bits [Y 31.0706 dB    U 40.6075 dB    V 41.8990 dB] [ET     2 ] [L0 ] [L1 ] [MD5:416c85727cc4f84f6f9f521471453576,22a07b3f2c6b55e911517a3d28f9ea66,a583438bf83ec3673f3ab9eabba26427]
    POC   21 TId: 0 ( I-SLICE, nQP 42 QP 42 )      43048 bits [Y 28.0718 dB    U 39.1126 dB    V 40.5276 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5fa37d16b7e20c0b41a9226b262f75fb,71803cb85f69cfba2ae34711193a9f03,b46be23bb0ecd9f6d99a7efa7b181e02]
    POC   22 TId: 0 ( I-SLICE, nQP 42 QP 42 )      22440 bits [Y 30.9226 dB    U 39.5580 dB    V 41.6625 dB] [ET     2 ] [L0 ] [L1 ] [MD5:55e348e9b7e74834bc6a9f65ef4ac118,772a29404f4a38ff7dafceff7455d3f1,54f1d1946219075b005fc3548bb9642c]
    POC   23 TId: 0 ( I-SLICE, nQP 42 QP 42 )      84648 bits [Y 26.1314 dB    U 37.4258 dB    V 36.7209 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6d1b307423487ceb04a1abc8eb96b2cc,674b8fc1ae72b7ea0143e97aff8e5be9,7e87ada1911c3fbf05b67ddf2af7807f]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    2239.1200   29.0659   39.3347   39.3731   30.0804  

# 41
    POC    0 TId: 0 ( I-SLICE, nQP 41 QP 41 )      39888 bits [Y 28.9497 dB    U 40.3606 dB    V 38.5337 dB] [ET     2 ] [L0 ] [L1 ] [MD5:52a10b81bb77e2fdfce4a2bd07127d31,cb64a565eea3cdd9e11cc84873fa8d6b,5ddbaf79791968f9dd6229e720eeb47f]
    POC    1 TId: 0 ( I-SLICE, nQP 41 QP 41 )      92720 bits [Y 26.7937 dB    U 36.6214 dB    V 36.9066 dB] [ET     3 ] [L0 ] [L1 ] [MD5:efc1c596d08a403b8bff923946d2de20,459276f5a0bba8669820ed5e4a3c31e9,3641989ca40596f6570019c5e5d9c423]
    POC    2 TId: 0 ( I-SLICE, nQP 41 QP 41 )      24096 bits [Y 30.9592 dB    U 42.2378 dB    V 37.4456 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f0f54376046b421719f46609397f8f55,3ff128e6bb3b413abda3253e3c7d6bf1,797e86ad36a6e4b71f6094b9915fff65]
    POC    3 TId: 0 ( I-SLICE, nQP 41 QP 41 )      29928 bits [Y 31.2095 dB    U 40.3386 dB    V 41.2702 dB] [ET     2 ] [L0 ] [L1 ] [MD5:24279652b0783ac5b2e3aec82d7ebc1f,db67d1e1be81340e4cbfcfbe464896d3,221faa867ba3a8addfacec97c0eedd54]
    POC    4 TId: 0 ( I-SLICE, nQP 41 QP 41 )      47112 bits [Y 28.0121 dB    U 40.3932 dB    V 40.9458 dB] [ET     2 ] [L0 ] [L1 ] [MD5:14839d522fbeac7f04fe69a30652202a,868829489c29de1591b0b373977d1d15,d21697942fa3d550cfd8ecc6b2d8e543]
    POC    5 TId: 0 ( I-SLICE, nQP 41 QP 41 )      18504 bits [Y 31.8536 dB    U 41.3881 dB    V 41.3325 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5a0f91037e94d253c41978d97c3c2645,883a54bf84d0b1858f8edba619f519c9,507fd2267cc554c4f39b1a7711f99f81]
    POC    6 TId: 0 ( I-SLICE, nQP 41 QP 41 )     105296 bits [Y 25.0114 dB    U 37.5899 dB    V 39.8261 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e2efbb4410b2064ba30630cb03028e6e,eab26e01f50380df14050105017d4402,98ce2caafb9891772d05dfbd49ae04dd]
    POC    7 TId: 0 ( I-SLICE, nQP 41 QP 41 )      39608 bits [Y 31.0780 dB    U 38.1700 dB    V 39.8908 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a85791962e3d38a06c43bc59228f40ae,7b0a468c6a01ed2323654f3fa409ef97,14ccbebf0041e85bd1dc1b5d478ac0b7]
    POC    8 TId: 0 ( I-SLICE, nQP 41 QP 41 )      19712 bits [Y 32.0281 dB    U 40.5052 dB    V 41.4877 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0e003f04ef3f23416faf55ad15700478,34ff9f1a88a9ffac27561150dffd573a,acffa2e841b5368a3726edb24aab5fa8]
    POC    9 TId: 0 ( I-SLICE, nQP 41 QP 41 )      34368 bits [Y 30.5210 dB    U 41.6022 dB    V 42.1914 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ba0fc82a3c08945d07aa7d769676e1ae,ba6e1828e2bef9275333733f5b44ab98,af4e988f543d9b338d669c426e0eddce]
    POC   10 TId: 0 ( I-SLICE, nQP 41 QP 41 )      23904 bits [Y 30.0460 dB    U 42.0004 dB    V 43.6090 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4dcd2ced10582c38bee1e693484c7c73,7c5e509bff9484d13cace1f46e5f601d,936adae4b1e583e5cf0a6b9247027220]
    POC   11 TId: 0 ( I-SLICE, nQP 41 QP 41 )      16176 bits [Y 31.5132 dB    U 40.6918 dB    V 37.5505 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ee02a30e69e40606d117633c5e43d4fc,01a225dccb1066ecc820f366b745bce7,3d08910da8d38d0e55b6d36c3d38924b]
    POC   12 TId: 0 ( I-SLICE, nQP 41 QP 41 )      59720 bits [Y 27.9030 dB    U 37.2411 dB    V 37.4373 dB] [ET     2 ] [L0 ] [L1 ] [MD5:868c771196b6bf45ef3c1aa64adced51,977c880ba647ea089efba8f4389ab685,e256a7c00ffe2667c309c080ac75b7c2]
    POC   13 TId: 0 ( I-SLICE, nQP 41 QP 41 )      61976 bits [Y 26.5454 dB    U 41.1407 dB    V 39.9903 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ab2b7b0904cea6b32d958ea832b24091,64fad3286026dec2ca1c9e16e6952ab6,c5540c88f6f3ce1f97f33642c100cbb2]
    POC   14 TId: 0 ( I-SLICE, nQP 41 QP 41 )      23392 bits [Y 31.2772 dB    U 41.0066 dB    V 38.0208 dB] [ET     2 ] [L0 ] [L1 ] [MD5:647eb55cac31bfa25e8c521d4fdbf049,7ec26bc11ac4b1642debf7d8f49b5c72,b7fe0199427948e590befeff873e2c49]
    POC   15 TId: 0 ( I-SLICE, nQP 41 QP 41 )      66488 bits [Y 27.5984 dB    U 37.6388 dB    V 38.8282 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d200db8417dfeb582d8e6ca875ff53bc,49bdd1e8f049043a32517bb0c6f466af,2fb75eee953dfc13883c23d3883eb386]
    POC   16 TId: 0 ( I-SLICE, nQP 41 QP 41 )      60472 bits [Y 27.4456 dB    U 37.6921 dB    V 38.1498 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bcde8283f926af8dd05a4fcfdd307e3d,7f4945f7f2abdbe30b8e81ddc7bd4fb9,e06590f8a5775df7c09ac8706aa66b09]
    POC   17 TId: 0 ( I-SLICE, nQP 41 QP 41 )      34624 bits [Y 29.6382 dB    U 40.6823 dB    V 41.3196 dB] [ET     2 ] [L0 ] [L1 ] [MD5:609ecb5ef1f8cf7ff525cc56da5278fa,6711d55287d3cf80ecda11b52aa4f64c,41bda81a61002749fde18baf17810fee]
    POC   18 TId: 0 ( I-SLICE, nQP 41 QP 41 )      32728 bits [Y 29.1903 dB    U 38.3874 dB    V 38.7907 dB] [ET     2 ] [L0 ] [L1 ] [MD5:92d06636b88904205f11a586243fe728,63019608356bbe473fe70382dd94310f,56541ddbfcb117858ab666da2207693c]
    POC   19 TId: 0 ( I-SLICE, nQP 41 QP 41 )      25632 bits [Y 33.4264 dB    U 39.9177 dB    V 39.6506 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ebc9682d118bd5c5aca53b650cc37cb8,fc857bdc62bc16a1195724fa27a5837f,2a85f972085202eab28aefc6ea0a7d6e]
    POC   20 TId: 0 ( I-SLICE, nQP 41 QP 41 )      27512 bits [Y 31.5758 dB    U 41.0156 dB    V 42.4675 dB] [ET     2 ] [L0 ] [L1 ] [MD5:03cf33869a0d150c19c35888f6d0dbdf,2fe8973616555b7cb11de4818528c151,945b20a0157216b9d678ae038654b4da]
    POC   21 TId: 0 ( I-SLICE, nQP 41 QP 41 )      50888 bits [Y 28.5716 dB    U 39.4759 dB    V 41.0021 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b5fd27dc520b11011bf17e199c13fb75,993a82b9d7d06025294212deeefd8f8d,2bec4f767a46905390f42e5e5bab34e1]
    POC   22 TId: 0 ( I-SLICE, nQP 41 QP 41 )      25432 bits [Y 31.4017 dB    U 40.0793 dB    V 42.0643 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7e22a8d60fb05cd11459195f935d799c,e86caa8080095525d3ecacdd06fc46f5,6018b8ff67ee6f15307cc591e56bdeb4]
    POC   23 TId: 0 ( I-SLICE, nQP 41 QP 41 )      97904 bits [Y 26.7256 dB    U 37.8106 dB    V 37.3740 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6956e3e20ebad73654e5ff6a8690dbd6,78e47f418e3c282f75506f4a6689f74f,951bcc8703408da3b41d408e28eeb2ec]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    2645.2000   29.5531   39.7495   39.8369   30.5843  

# 40
    POC    0 TId: 0 ( I-SLICE, nQP 40 QP 40 )      46752 bits [Y 29.4516 dB    U 40.2596 dB    V 38.4527 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f38db7a416ab31baed3d0969e4cf275b,0660dd1603ff01d6d41e1fd70e23d0ab,6c87736d6ecc72010123d9729825db33]
    POC    1 TId: 0 ( I-SLICE, nQP 40 QP 40 )     106128 bits [Y 27.3732 dB    U 36.6059 dB    V 36.8702 dB] [ET     3 ] [L0 ] [L1 ] [MD5:bc10543ed27a06ee0ad645ab6ca6079d,0e3145ab7a075bf05fbf11c83cee25ad,a05ebb8985cc09dcd464627f633bcfac]
    POC    2 TId: 0 ( I-SLICE, nQP 40 QP 40 )      27784 bits [Y 31.3986 dB    U 42.4659 dB    V 37.3097 dB] [ET     2 ] [L0 ] [L1 ] [MD5:da5a2824f7ea61d1c2b88fdc5797a7b8,13c4d6edc46736f0f93858c712e66752,8f4461bc23f102961cccc35bddc711a9]
    POC    3 TId: 0 ( I-SLICE, nQP 40 QP 40 )      33744 bits [Y 31.7358 dB    U 40.3364 dB    V 41.0022 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cdbbf5b3aa1a8dcec461a6a1cb63eff6,f62bb7c09319a4e6fa97124fb155dcba,d2261b54ab32a199e0110014cbb8808d]
    POC    4 TId: 0 ( I-SLICE, nQP 40 QP 40 )      55616 bits [Y 28.5133 dB    U 40.3215 dB    V 40.9363 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4ab6d637cee62ab7998c65df79ffcd50,4c53d49da0269c84ef4e5d296a6a0556,2b406fea2bffc3c235578528418ca8c0]
    POC    5 TId: 0 ( I-SLICE, nQP 40 QP 40 )      21728 bits [Y 32.3050 dB    U 41.6929 dB    V 41.1779 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4a94688bce339e04798711294d0546b2,4d7ba373c97abfb2a149b1d178acb175,b7168ad6e858d977eac73321b6011fbb]
    POC    6 TId: 0 ( I-SLICE, nQP 40 QP 40 )     126936 bits [Y 25.6190 dB    U 37.6326 dB    V 40.0134 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1366c5553ffaa5bd8bb9d114cb373739,4f12c2ca6aa3a61c5e15ba842de4ca18,d8968771d1d68f4a890c93faa9d4fd61]
    POC    7 TId: 0 ( I-SLICE, nQP 40 QP 40 )      44968 bits [Y 31.7365 dB    U 38.0617 dB    V 40.0176 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dee3f3059611f4147e4b63fac0d3ed36,9a38bf193eebddf2ca5b412684bbad05,e1f94caa091112abf02758d2700e9f67]
    POC    8 TId: 0 ( I-SLICE, nQP 40 QP 40 )      22528 bits [Y 32.5205 dB    U 40.3630 dB    V 41.3232 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2165d8c3c87334b9310d3493180ddd61,008743f2258ceaa6ba22c7406aa9ed2e,9ad01dbaca122930ec600f21728ae117]
    POC    9 TId: 0 ( I-SLICE, nQP 40 QP 40 )      39992 bits [Y 31.0723 dB    U 41.7006 dB    V 42.2033 dB] [ET     2 ] [L0 ] [L1 ] [MD5:334974aa9304f95b9c68f020b134fcc6,509a96da2218beed49717e253bd0b41c,12b0a4cc4cbb7be63fefc8e284cc5083]
    POC   10 TId: 0 ( I-SLICE, nQP 40 QP 40 )      29096 bits [Y 30.4847 dB    U 41.9646 dB    V 43.6211 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c7db6c3cfd6b2db395258f449663a008,81566d0cc5ed15f0f33e9e93a817ac66,7c1c72260123dcf5c7b615fba4c17a83]
    POC   11 TId: 0 ( I-SLICE, nQP 40 QP 40 )      18912 bits [Y 31.8861 dB    U 40.7440 dB    V 37.5316 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f0da9ae7bb3d2cef27aa86cf11a0631d,831a45f5cb4ddc343206cff065ed30f8,7cf0eec905e67c668778fd5f925a41a0]
    POC   12 TId: 0 ( I-SLICE, nQP 40 QP 40 )      68952 bits [Y 28.4034 dB    U 37.2257 dB    V 37.5319 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8b44f9d33c7662618a5e490bacf38ff4,480add39effd024cda0922e3ec42a14a,a0521541bc337bfc7c644501a05dedd9]
    POC   13 TId: 0 ( I-SLICE, nQP 40 QP 40 )      75776 bits [Y 27.1128 dB    U 40.9199 dB    V 39.9846 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7361db43c2d7622ecdfad5d87cb9cf17,34f2f3722f197e4ff28e74d6eaecb880,7b34ed98ad66b76923dafd49d69e781c]
    POC   14 TId: 0 ( I-SLICE, nQP 40 QP 40 )      26792 bits [Y 31.6705 dB    U 41.0144 dB    V 38.0770 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8ed4b3e328540ef7e4c47fbd12eb96c6,ce16d7b9b107df11cd4456a9e8cf4f71,977a067851793d8d2444c73a42bd7f54]
    POC   15 TId: 0 ( I-SLICE, nQP 40 QP 40 )      78880 bits [Y 28.2324 dB    U 37.7616 dB    V 38.9162 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ce5cb0e660bd6446ab7fd42168f7f4f2,c76391369794d3f1853cdd942e4d0049,f547b50144608247911bcfd1fe54037c]
    POC   16 TId: 0 ( I-SLICE, nQP 40 QP 40 )      72520 bits [Y 28.0335 dB    U 37.5902 dB    V 38.1756 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4be651442b81ef7d8cb294df6792ef27,39748ad7d030d955a994a6cb2e14a2c8,9deadac81723aa8cbeb17e5b1e901263]
    POC   17 TId: 0 ( I-SLICE, nQP 40 QP 40 )      39824 bits [Y 30.0663 dB    U 40.7647 dB    V 40.9849 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1e08e7ecbdd89b3d064d5654e493f202,66976d1204a64344fefa274a93ef6e6e,d3a1776e411ff6fce853b05994198d30]
    POC   18 TId: 0 ( I-SLICE, nQP 40 QP 40 )      38368 bits [Y 29.6230 dB    U 38.4295 dB    V 38.7185 dB] [ET     2 ] [L0 ] [L1 ] [MD5:41edfaa45aa71f17c57771528fa59f7b,6bd9575b05a041f317dbfabb51470c1b,4be314d4948db9336157bebbf8ec180c]
    POC   19 TId: 0 ( I-SLICE, nQP 40 QP 40 )      27928 bits [Y 33.9096 dB    U 39.9236 dB    V 39.5721 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3b60e51d6acc4b1db9762a60915f676c,33b139956d56058c2fdfaefcd0ba6a22,d1ef1261a12b8272ae2576f163938348]
    POC   20 TId: 0 ( I-SLICE, nQP 40 QP 40 )      31384 bits [Y 32.1479 dB    U 41.1546 dB    V 42.5452 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4eb112c96c35f56e36ffa909c291ab2f,67a6c5f55cc462f3a1bebe18fbe9e57f,1730207cc8dfdb2538030fd62e2df6c2]
    POC   21 TId: 0 ( I-SLICE, nQP 40 QP 40 )      60288 bits [Y 29.1564 dB    U 39.5315 dB    V 40.9621 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f12f79f51e317646c3fe094bdf0bbb34,7949c2ea5088cc5614caaa59493461e2,579e3a44638b3e3a4a69ba5a045abc44]
    POC   22 TId: 0 ( I-SLICE, nQP 40 QP 40 )      29200 bits [Y 31.9274 dB    U 40.3344 dB    V 42.2482 dB] [ET     2 ] [L0 ] [L1 ] [MD5:334646f840354e130612afc97484d4df,782ec35fb11a60f6b2b8a2d4638d583f,37ec640b2929a7e3bc44e4bde3dc7e45]
    POC   23 TId: 0 ( I-SLICE, nQP 40 QP 40 )     112432 bits [Y 27.4054 dB    U 37.7381 dB    V 37.1460 dB] [ET     2 ] [L0 ] [L1 ] [MD5:330e23bf9c0b8bfccfc226c3e902bc59,4c7d5f40f024c55ce496c37303290ce7,fee2c11811fa41ef85d4fe64365f9b3e]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    3091.3200   30.0744   39.7724   39.8051   31.1021  

# 39
    POC    0 TId: 0 ( I-SLICE, nQP 39 QP 39 )      56032 bits [Y 29.9888 dB    U 40.7654 dB    V 38.8399 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d5664f2fd8af24a5826873829703aef0,4c61e7804f661ab47386c60737044e04,055ba352c22899b52a9444a31e5f7272]
    POC    1 TId: 0 ( I-SLICE, nQP 39 QP 39 )     124024 bits [Y 28.0546 dB    U 37.0404 dB    V 37.2821 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8a24676107c54d3232ddd8340db1426a,f0e3ffe74f56b3c4ff06f94127bc3335,80906f6f5ca7ff537b3996e2e2da3d55]
    POC    2 TId: 0 ( I-SLICE, nQP 39 QP 39 )      32640 bits [Y 31.8412 dB    U 42.4914 dB    V 37.8601 dB] [ET     2 ] [L0 ] [L1 ] [MD5:96a32083f2cae265cd510d7d0aef21b3,a4562cb582ec7a9d07f3d933d24a4754,e83ef071704be726c2f7f277c6cab352]
    POC    3 TId: 0 ( I-SLICE, nQP 39 QP 39 )      39472 bits [Y 32.3083 dB    U 40.6163 dB    V 41.4162 dB] [ET     2 ] [L0 ] [L1 ] [MD5:44c95645c082c07e8f8b0a924cb0e6b5,6b5ab91968695bbb5bcdcff5daaaae2d,475d7e2d9d193e0418ed548a96083af8]
    POC    4 TId: 0 ( I-SLICE, nQP 39 QP 39 )      67480 bits [Y 29.1082 dB    U 40.9611 dB    V 41.2538 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e49c2e578685d32ac37ee6903e7ec1c1,322e5892a0a2b1b9003a2758cce62e62,621688c526cf30bd9305a20ab3db77a1]
    POC    5 TId: 0 ( I-SLICE, nQP 39 QP 39 )      24688 bits [Y 32.6958 dB    U 42.0317 dB    V 41.7983 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1290c3ccfe9a97d1f1aaa312a156e5c1,730a5cd66295635340a79172166f8fce,44736bac94ba453724154ea21e8c7234]
    POC    6 TId: 0 ( I-SLICE, nQP 39 QP 39 )     152528 bits [Y 26.2428 dB    U 37.8320 dB    V 40.2831 dB] [ET     3 ] [L0 ] [L1 ] [MD5:558fbc78e687c4476310eca4c77a71a1,f2771e71f5998a21938aab6603641101,4b4c17a547fab25cd2908ccc9c0c6a98]
    POC    7 TId: 0 ( I-SLICE, nQP 39 QP 39 )      51280 bits [Y 32.3504 dB    U 38.8445 dB    V 40.4690 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d83fb882e9b3d65b032495886555f896,48beac38d733a8379c36d3c8471c9e7b,d53f19c732f2ea6e9a8973f71319bedb]
    POC    8 TId: 0 ( I-SLICE, nQP 39 QP 39 )      26800 bits [Y 33.0363 dB    U 40.7363 dB    V 41.7119 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c4945c888d583540c33cbaeab7ea90aa,ce033220e30d9b63b4c3cfc5bc153ad5,3b04c52c06acfe82a8343d3692fbda2b]
    POC    9 TId: 0 ( I-SLICE, nQP 39 QP 39 )      45440 bits [Y 31.5919 dB    U 41.8618 dB    V 42.8038 dB] [ET     2 ] [L0 ] [L1 ] [MD5:db1f45068e6a1263fbb368e290c7f9f1,f9a5fe84998edd8a293bf5acb6233b37,1ebb8782a666556058b27c83af14ec24]
    POC   10 TId: 0 ( I-SLICE, nQP 39 QP 39 )      35488 bits [Y 31.0239 dB    U 42.3200 dB    V 44.0046 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8a765fb79837cbba8c0c824593d1e270,724e4e7dc8b3b1aa50720b7bad954b43,e94c3c04e1d3e4f92d613a85ea3f2633]
    POC   11 TId: 0 ( I-SLICE, nQP 39 QP 39 )      22344 bits [Y 32.1737 dB    U 41.3357 dB    V 38.0381 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8c7cdf6d78c267539b37e986eac2997e,4d449d3b91eede5e1930ba588fb1be2c,dc64e42ca0af9b96b30a7ceaa37b315d]
    POC   12 TId: 0 ( I-SLICE, nQP 39 QP 39 )      81728 bits [Y 28.9686 dB    U 37.8729 dB    V 38.0933 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8c7dec814a386885f798a2ea7ab44b29,3f58d18431f36211f43b3aae79af9061,a0c647e81ab4a0de6bd1416902ea1a19]
    POC   13 TId: 0 ( I-SLICE, nQP 39 QP 39 )      91256 bits [Y 27.6467 dB    U 41.0305 dB    V 40.1747 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a408c672250bf8cd056cea14acc7e422,0e0cef739ce9e59d147449107f84bb48,37356a4627af4b120cfcc4f471565509]
    POC   14 TId: 0 ( I-SLICE, nQP 39 QP 39 )      31384 bits [Y 32.1248 dB    U 41.5214 dB    V 38.5523 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9fc4ba1e8f3857814e47f0074d705e41,f9371ca941e82908882ca14d0a299e57,fdde851c49ee899166009de1eb6f58f1]
    POC   15 TId: 0 ( I-SLICE, nQP 39 QP 39 )      93496 bits [Y 28.8634 dB    U 37.9269 dB    V 39.2879 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6bf4c99cef6fc68a9a91b3a8415bd742,ad09ced3eaace93f824f9bd660a0d72e,3bda82ea21973f52404b3d65b34c4243]
    POC   16 TId: 0 ( I-SLICE, nQP 39 QP 39 )      88896 bits [Y 28.6979 dB    U 38.0605 dB    V 38.4953 dB] [ET     2 ] [L0 ] [L1 ] [MD5:fe6b4b56df8624c1a38d05bc7196db4d,1e2563e4225e04561b8f39f33050577c,0cd01ce80579e9c559b43aac5c24b948]
    POC   17 TId: 0 ( I-SLICE, nQP 39 QP 39 )      46632 bits [Y 30.5484 dB    U 41.1584 dB    V 41.7372 dB] [ET     2 ] [L0 ] [L1 ] [MD5:40432e3b735f6ce88118b5928ec302e8,1d5d430265e276a876657cb5477cb94a,f56c6219c5cd824fc5f14cf5891cf3d8]
    POC   18 TId: 0 ( I-SLICE, nQP 39 QP 39 )      46600 bits [Y 30.0803 dB    U 38.7935 dB    V 39.1879 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a6cc57d2629880824a5ff84efd88ca51,cbf71e94aa0f7894dc55837941544d54,868d5e54b102eec36ecb672a99cddf58]
    POC   19 TId: 0 ( I-SLICE, nQP 39 QP 39 )      31960 bits [Y 34.4966 dB    U 40.2880 dB    V 40.1991 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ee5da15435828499016f4f0a4f484811,25c195e79ada5123b32d5751dc31b1dc,a7f676c19491266611ad766420ccf661]
    POC   20 TId: 0 ( I-SLICE, nQP 39 QP 39 )      36320 bits [Y 32.7138 dB    U 41.5751 dB    V 42.8398 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a69ad27be2ca64d415d9493c8b90693c,ea502a0de804839f6a9a91e073205355,8fa9960ee926b4a8f8b222decb61fa50]
    POC   21 TId: 0 ( I-SLICE, nQP 39 QP 39 )      70864 bits [Y 29.7618 dB    U 39.8834 dB    V 41.2699 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6c65ee6b3f9d55e801b04651e55f5bd3,262bfec229b45c184748de9dd1571228,751d71a7346b19d2fa0c877d2050e7ad]
    POC   22 TId: 0 ( I-SLICE, nQP 39 QP 39 )      33144 bits [Y 32.3833 dB    U 40.4327 dB    V 42.5021 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6cd8dc7a5f7e54e59fe966365b4d9039,32910252c924819fb8e16b030930b258,85bc6656bae33c8a708a9150617ea5bd]
    POC   23 TId: 0 ( I-SLICE, nQP 39 QP 39 )     128704 bits [Y 28.0628 dB    U 38.4528 dB    V 37.8601 dB] [ET     2 ] [L0 ] [L1 ] [MD5:de225c3ab19b270bb7488d74b7ccc25c,d304efa1c5aa7e7ff74c9189338566f0,e388f02ac1a237555b01f06d984dfa87]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    3648.0000   30.6152   40.1597   40.2484   31.6613  

# 38
    POC    0 TId: 0 ( I-SLICE, nQP 38 QP 38 )      65896 bits [Y 30.5675 dB    U 40.7473 dB    V 38.9107 dB] [ET     3 ] [L0 ] [L1 ] [MD5:22743fb9fc718bc928ecaa6c118b3642,623684f56a7954fd33196ea83cbb3251,937edc2fea53469c7e071a6c3e412718]
    POC    1 TId: 0 ( I-SLICE, nQP 38 QP 38 )     141256 bits [Y 28.6927 dB    U 37.1356 dB    V 37.3231 dB] [ET     3 ] [L0 ] [L1 ] [MD5:379850391342867a4ed96801ae99cac4,902fb2fe3348a7c0b34dfd4ddc9d489a,8a1ffc49a4d8b67310282666208618cd]
    POC    2 TId: 0 ( I-SLICE, nQP 38 QP 38 )      38280 bits [Y 32.2868 dB    U 42.6813 dB    V 37.8536 dB] [ET     2 ] [L0 ] [L1 ] [MD5:58087ecfe82e937ad7be0aa63da16a59,3e40255f1dab1ff2d8041cab2dc054a9,50acf75489ed69f19046a808339a18b5]
    POC    3 TId: 0 ( I-SLICE, nQP 38 QP 38 )      44960 bits [Y 32.9068 dB    U 40.6281 dB    V 41.2865 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9f04a0b370549dde2f82ea359ba408d9,ca12d944e2cf03beee60bbcac39385ea,d30afffe7fc42a6a7163ee07ad5d5233]
    POC    4 TId: 0 ( I-SLICE, nQP 38 QP 38 )      80080 bits [Y 29.7109 dB    U 40.8820 dB    V 41.3734 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8e10f82efb0615aea17baddcdd769686,a0a634537c151dcb437c17077e4d7e9c,8f5735222c5144cf484802b7b50816be]
    POC    5 TId: 0 ( I-SLICE, nQP 38 QP 38 )      29096 bits [Y 33.2029 dB    U 42.0531 dB    V 41.8266 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cf037f17a1c3b2f5a9f8c344c3e190a7,cf959277081e4ca2b49156fa3770fb95,a31b7e370ee4e08de0fd11c5dda03c15]
    POC    6 TId: 0 ( I-SLICE, nQP 38 QP 38 )     180328 bits [Y 26.8793 dB    U 37.8357 dB    V 40.1489 dB] [ET     3 ] [L0 ] [L1 ] [MD5:aea6a6609cfc59e2bd577c7eb2bfa883,887afe430df2bcf65cab11d1a0b67a59,3ff15fbe7c73b817148780795c7dab8a]
    POC    7 TId: 0 ( I-SLICE, nQP 38 QP 38 )      57392 bits [Y 33.0254 dB    U 38.6729 dB    V 40.3074 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a8b7c5de9b1970239540697257a9ea3a,415301ece5e489c86487b107479d54d8,82243b1ea1357e9e6df00081b965b5fd]
    POC    8 TId: 0 ( I-SLICE, nQP 38 QP 38 )      31680 bits [Y 33.6251 dB    U 40.6584 dB    V 42.0448 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8c51c5d70ebca0bcb0a98c8f31ea4a10,bcd26b51d2254ade29de8d5fde2c68ed,f578c2d4eb6afd9f5ca7087813588235]
    POC    9 TId: 0 ( I-SLICE, nQP 38 QP 38 )      51760 bits [Y 32.1344 dB    U 41.9801 dB    V 42.7122 dB] [ET     2 ] [L0 ] [L1 ] [MD5:88ae12de9b58549b85fca9c516366629,1d60130124dcddcec4f1515a2f440c81,abfa113b036df79bc1d8fa05d78ed72e]
    POC   10 TId: 0 ( I-SLICE, nQP 38 QP 38 )      42400 bits [Y 31.5269 dB    U 42.2705 dB    V 43.8268 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f909fb51ac3f71aea5d0cdd1aa9568cd,ba378f1d1519cd43ad295113b2f513da,397bf0f52d217b728e99e6e0232d35ee]
    POC   11 TId: 0 ( I-SLICE, nQP 38 QP 38 )      27184 bits [Y 32.6403 dB    U 40.9293 dB    V 38.1324 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dad1badf80c175e532a24b5dc03519b4,087ddee03b06d4bc012736312d34e57a,910ed1e9a33a01e08cdb5e9b7d76e6ab]
    POC   12 TId: 0 ( I-SLICE, nQP 38 QP 38 )      94224 bits [Y 29.5049 dB    U 37.7393 dB    V 38.0024 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e25fab2b5bc85abba9b6dd3a4e415e1d,60205d31c71b7b45142f397eb24d0a28,f433b2e2cd573ca8aefc69ea5a8077ab]
    POC   13 TId: 0 ( I-SLICE, nQP 38 QP 38 )     109656 bits [Y 28.2165 dB    U 41.1453 dB    V 40.5013 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9c74fa28a1758cb980f4f6fc65913aa4,78de1595eb90ed5a1d6a4c64e52deec2,bf43c03462a980d1c087b1be4829681f]
    POC   14 TId: 0 ( I-SLICE, nQP 38 QP 38 )      36552 bits [Y 32.6503 dB    U 41.3253 dB    V 38.5634 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a72cc81c017e8b96abc2597d7122c335,760edc031219ceebd50c54c96ab93181,f3ecefd256d1b107f6f7e37e2f8a59a7]
    POC   15 TId: 0 ( I-SLICE, nQP 38 QP 38 )     108152 bits [Y 29.4911 dB    U 37.8243 dB    V 39.0094 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0aa8853cf2e5699865da90d119d5857e,44d54cea8ac90b53f8a3ccf40ad941c0,1edbd60223e0cd25608d66b29664fe4a]
    POC   16 TId: 0 ( I-SLICE, nQP 38 QP 38 )     103080 bits [Y 29.2944 dB    U 37.8639 dB    V 38.4408 dB] [ET     2 ] [L0 ] [L1 ] [MD5:257516bb5b611f97a543daeaea3a0948,20955e2f70e86a89a7b3e94c54bc78ef,6ff94e121050e8207864666b48db4833]
    POC   17 TId: 0 ( I-SLICE, nQP 38 QP 38 )      53968 bits [Y 31.0185 dB    U 41.3354 dB    V 41.8093 dB] [ET     2 ] [L0 ] [L1 ] [MD5:389d0f10f719718645cbdf01c3390f59,c32a667dbda8b5e606484fdbf83c3006,810f333b6418fb461be97e35dd464b7a]
    POC   18 TId: 0 ( I-SLICE, nQP 38 QP 38 )      54608 bits [Y 30.5440 dB    U 38.8005 dB    V 39.0722 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5d888b110982f287563b3e4b8a0679ca,8895275e5ab8e8cc38915406e5145e45,580ac51d6d122a4037a367686da8f9bf]
    POC   19 TId: 0 ( I-SLICE, nQP 38 QP 38 )      35024 bits [Y 35.0116 dB    U 40.4033 dB    V 40.0667 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6385c958a00cc66dc281d9cfc36b839c,5f62426bf33c0241ec419556bd303bee,16694bbdf2c780a62ee6d3d8872203cf]
    POC   20 TId: 0 ( I-SLICE, nQP 38 QP 38 )      41440 bits [Y 33.3378 dB    U 41.4709 dB    V 42.9343 dB] [ET     2 ] [L0 ] [L1 ] [MD5:af89987a3740fec64520fbee9c0b7ca5,717da257a75076c32e361205d1ac75c8,c86914f74076e6bdff3c3a01fce197f1]
    POC   21 TId: 0 ( I-SLICE, nQP 38 QP 38 )      82088 bits [Y 30.3374 dB    U 39.8275 dB    V 41.4223 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f9ef398303cfb7ba0c3995502214c7e9,165fda936c1f536603173e4bd32648b4,ffc1eaeaa4cf83874ff59d34e58010bf]
    POC   22 TId: 0 ( I-SLICE, nQP 38 QP 38 )      38000 bits [Y 32.8691 dB    U 40.6042 dB    V 42.4574 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5fb740ff710db732be67d03ce5cc1a34,c1c62bdf54ec1b4e840deb08bd51a11c,a791daf4a4ba3d4eded954f4ec9947ea]
    POC   23 TId: 0 ( I-SLICE, nQP 38 QP 38 )     145360 bits [Y 28.7016 dB    U 38.2272 dB    V 37.7454 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0fb7effde47fb7bf2450b75a6cf515db,907c7599d8f516c3f7afeed2a4565c51,9206df5a1f1d75ffb05d33be30ac745b]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    4231.1600   31.1740   40.1267   40.2405   32.2014  

# 37
    POC    0 TId: 0 ( I-SLICE, nQP 37 QP 37 )      78120 bits [Y 31.1638 dB    U 41.3126 dB    V 39.3595 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b98205b5e1fddaa1c4c7944808a3c28b,8d5d2e2e5f1babc091022b74b501d722,5b5656be412815ed25caee8d228d215e]
    POC    1 TId: 0 ( I-SLICE, nQP 37 QP 37 )     161784 bits [Y 29.3910 dB    U 37.5804 dB    V 37.9473 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f280d1289630562dd2099781aba1a28e,8b6cb51aa7511773948aa1bb18c2725b,f3d33fb1dbe7320478e46ac0bee9a4d1]
    POC    2 TId: 0 ( I-SLICE, nQP 37 QP 37 )      44928 bits [Y 32.7675 dB    U 43.0851 dB    V 38.1769 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c177f9a2aa35239426340f2fd98f97e5,cf0c1cbd994145d6b43ab7a02c524c01,1fb6b2d499260fe42b027d129b66afa7]
    POC    3 TId: 0 ( I-SLICE, nQP 37 QP 37 )      52320 bits [Y 33.5345 dB    U 41.0675 dB    V 41.8331 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2fa1113f481a51fcb29f81547d58a18a,8fa53f788e1dd9cdf7787e39d343646d,7e0c2f222617c922a31263df054eaf9a]
    POC    4 TId: 0 ( I-SLICE, nQP 37 QP 37 )      96984 bits [Y 30.4037 dB    U 41.1749 dB    V 41.6917 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dc4a458b27acc245095243f5be7c02a3,d93c41fc713b07a00a4128c14fe8b6c7,70aa322e0c2d9451588d2655f6c6c412]
    POC    5 TId: 0 ( I-SLICE, nQP 37 QP 37 )      34304 bits [Y 33.6472 dB    U 42.5870 dB    V 42.2989 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9b784f10c27ff4b009dc231c1d98a623,6bf0f5e6db2111312e90bd14651bc3c9,d257b33a0800ee214a50a5c6b003b216]
    POC    6 TId: 0 ( I-SLICE, nQP 37 QP 37 )     215192 bits [Y 27.6115 dB    U 38.2175 dB    V 40.5628 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4fd530b4b3f4d68e8fbe9614eaa0402c,90b1a0ca6986df33def03998c7b69ace,fef6f66f6495ecf18ac4ffe8357f2653]
    POC    7 TId: 0 ( I-SLICE, nQP 37 QP 37 )      65376 bits [Y 33.6836 dB    U 39.4337 dB    V 40.8192 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0c01cc8de67e111278cdc2549eb4de0f,cf074fde5d5ff7a79b09341fac43bf8f,d051052534a001df0f228482e392c527]
    POC    8 TId: 0 ( I-SLICE, nQP 37 QP 37 )      36600 bits [Y 34.1522 dB    U 41.0299 dB    V 42.3644 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4ecfec96b3d076b9181b3e31e64a4791,f21a7b4643f5e1481078c497c7ac20f3,dcb4ebfc385534a52173a7bde0674775]
    POC    9 TId: 0 ( I-SLICE, nQP 37 QP 37 )      60400 bits [Y 32.7758 dB    U 42.1618 dB    V 42.9568 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8169701b4b58766370175e7cf7d043d3,17e5dfd36245c3362bdc2c3bdd330d48,f61273f64253dfb799004918d60d5084]
    POC   10 TId: 0 ( I-SLICE, nQP 37 QP 37 )      52136 bits [Y 32.0941 dB    U 42.9036 dB    V 44.3431 dB] [ET     2 ] [L0 ] [L1 ] [MD5:63a10161aae183117b9f91d83c324eaf,1bdea323dde766106f57687b192b1152,1f0821a7eaa74a87e6b3c84b524c1e78]
    POC   11 TId: 0 ( I-SLICE, nQP 37 QP 37 )      31648 bits [Y 32.9680 dB    U 41.2770 dB    V 38.5401 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1b6b486255a4c85ded05703292c5905b,8bd08db39f762c6a266a315316578bea,58659bf676d507d04b7c1e699c4f88c9]
    POC   12 TId: 0 ( I-SLICE, nQP 37 QP 37 )     110888 bits [Y 30.1169 dB    U 38.2547 dB    V 38.6790 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2ec570d62c090ddf0c7382ba59596f73,d108a9c9df7e39921f1aea561ee62254,0118c1b86f0d600d2a5be4807c7503d5]
    POC   13 TId: 0 ( I-SLICE, nQP 37 QP 37 )     131264 bits [Y 28.8448 dB    U 41.5653 dB    V 40.5944 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0fccd7208f8800b4afc191e66ee60690,365896aa39fe6924e83e8aa01112befa,3d4795a63eb7ce81d2134cb6993f33e9]
    POC   14 TId: 0 ( I-SLICE, nQP 37 QP 37 )      43936 bits [Y 33.1817 dB    U 41.8100 dB    V 39.0553 dB] [ET     2 ] [L0 ] [L1 ] [MD5:56c6df78be2425e74300bb5785be6beb,6a5bc182c2b91a8cb1899294bb33afb2,9ff63c6ee4b2fa5e4554eb4e3b5209e7]
    POC   15 TId: 0 ( I-SLICE, nQP 37 QP 37 )     127008 bits [Y 30.2249 dB    U 38.3160 dB    V 39.5447 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d1c9b458521ee48ba01b27b1816f5a65,409e9a92c08a5a9c1519068fa3ea0e0a,32f62b2c7a3d2672f1318c8b613e67d4]
    POC   16 TId: 0 ( I-SLICE, nQP 37 QP 37 )     123024 bits [Y 30.0022 dB    U 38.3956 dB    V 38.8373 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4d0d717bdc9fbad95f3258e798342ecf,11087759590f3907f448a3468045c16a,03d12c3f3d6e6c098ab236299324e6dc]
    POC   17 TId: 0 ( I-SLICE, nQP 37 QP 37 )      62624 bits [Y 31.4957 dB    U 41.3682 dB    V 42.0626 dB] [ET     2 ] [L0 ] [L1 ] [MD5:52ce60e5978ca59a6159c0739d051a47,c0854d1a169b701453198610ab8b9028,946f9dcd3aaa4f86f7eefc94c47ede1e]
    POC   18 TId: 0 ( I-SLICE, nQP 37 QP 37 )      67632 bits [Y 31.1520 dB    U 39.1048 dB    V 39.6162 dB] [ET     2 ] [L0 ] [L1 ] [MD5:799e1c897712b68111a04edab416aeed,97f89fb0a8efedccb9982b37aa78d674,187493da98c2226cc71944ad9c84a76c]
    POC   19 TId: 0 ( I-SLICE, nQP 37 QP 37 )      39816 bits [Y 35.5588 dB    U 40.9107 dB    V 40.5952 dB] [ET     2 ] [L0 ] [L1 ] [MD5:942e10f6e896d0cdc32a2db1006a030f,3acc995971a4b1964497f091e4389874,b404ed8a86389a806f8e16c585870119]
    POC   20 TId: 0 ( I-SLICE, nQP 37 QP 37 )      47576 bits [Y 33.9085 dB    U 41.8205 dB    V 43.4087 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b46326273ec9b9884c7e0f0e45490496,aec585c6a43cbd9dce196567bbedcd5f,0818fa48f30299df052405fbd8a81712]
    POC   21 TId: 0 ( I-SLICE, nQP 37 QP 37 )      95888 bits [Y 30.9738 dB    U 40.4028 dB    V 41.8302 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8d126c007b3b87d269828d1981c1fbee,a02981a2731b19d5a24686c750358e5e,027f3c05da7cec17beeb0530091729bf]
    POC   22 TId: 0 ( I-SLICE, nQP 37 QP 37 )      44200 bits [Y 33.3868 dB    U 41.0240 dB    V 42.9172 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3ac868fbbe36977db6d293158c78add4,16ca705fe402a2a0635f9dc8b97926ef,5b0bc766517c873f5163095f83d02cfd]
    POC   23 TId: 0 ( I-SLICE, nQP 37 QP 37 )     165040 bits [Y 29.3973 dB    U 38.8537 dB    V 38.5283 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a10b1297c9676f5d65b15ddf8b7935c3,44947d5a89080e4a529ae5c4758dd640,f5fa1a1d490a20c117a2e25d2e0465f4]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    4971.7200   31.7682   40.5691   40.6901   32.8164  

# 36
    POC    0 TId: 0 ( I-SLICE, nQP 36 QP 36 )      91128 bits [Y 31.7707 dB    U 41.2057 dB    V 39.2864 dB] [ET     3 ] [L0 ] [L1 ] [MD5:52a0545c36f12f9372fd717f0f188185,83ca25a5d154a587536b6c111ebb0cc4,1f93eb741a49beeea53ff86fe31ef9a5]
    POC    1 TId: 0 ( I-SLICE, nQP 36 QP 36 )     185208 bits [Y 30.1561 dB    U 37.6040 dB    V 37.9242 dB] [ET     3 ] [L0 ] [L1 ] [MD5:9622c021bcb312f39f2cac8764ce8678,a5519a51f8f146010d362c74c207b8cc,2525539c7ea4ed42e80289bdf4658b82]
    POC    2 TId: 0 ( I-SLICE, nQP 36 QP 36 )      51736 bits [Y 33.2092 dB    U 43.0514 dB    V 38.1906 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3d674be09115b33dc621b8972142571a,7bf1c919da2cbcd10bdad8af1d772f95,ae86770d2f6375f130309ba0e7555273]
    POC    3 TId: 0 ( I-SLICE, nQP 36 QP 36 )      59104 bits [Y 34.1427 dB    U 41.0226 dB    V 41.7812 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8e28fc1ff9a333fe70b65c56d43316e9,f18c1104fa8ef363470b3c9561ed2760,b06c23d05ce359fdfd2e54e2a3c13b33]
    POC    4 TId: 0 ( I-SLICE, nQP 36 QP 36 )     114256 bits [Y 31.0658 dB    U 41.2052 dB    V 41.5654 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4942c001c70f460eab15e9c7f441f5c5,a94c98c94c9117c43bae0da4b99c0893,2e452c9307b7d67b6af59b29c3c14b9a]
    POC    5 TId: 0 ( I-SLICE, nQP 36 QP 36 )      39920 bits [Y 34.1916 dB    U 42.3886 dB    V 42.1365 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c01258a0150ea42311491a50344b9b45,62a58da35cbf33f84ed0c43fa8fc1157,52f0219380f8b210eebf949c37de2423]
    POC    6 TId: 0 ( I-SLICE, nQP 36 QP 36 )     251472 bits [Y 28.3349 dB    U 38.1280 dB    V 40.4553 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ef7b7eee1c7ddc1e0441fa9d991adf4b,7afc9b69ce6e701224e54558368747e4,8c2680f95fba9a12d79e259024e11468]
    POC    7 TId: 0 ( I-SLICE, nQP 36 QP 36 )      73232 bits [Y 34.3712 dB    U 39.2256 dB    V 40.6892 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2dcf61d78fa79626882190ae850e834a,3c72834a2a0fc929eba2785f728055de,65eb8b1ba121049e41832a555f919e29]
    POC    8 TId: 0 ( I-SLICE, nQP 36 QP 36 )      43152 bits [Y 34.7794 dB    U 41.0331 dB    V 42.2666 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7c79323dcb4441b53ad0ace7b932a76e,7be35a4733f5aa9b817ba406f1772acc,a08ed8e1ba1c40dec4dd1c393d30fbf1]
    POC    9 TId: 0 ( I-SLICE, nQP 36 QP 36 )      69328 bits [Y 33.3853 dB    U 42.1443 dB    V 42.7636 dB] [ET     2 ] [L0 ] [L1 ] [MD5:af4daafe845eba96b0990774cc9cd407,e4713187eb0a8e8385706a46e8014e5e,2515423490d1f593c3b8c68c32b5faea]
    POC   10 TId: 0 ( I-SLICE, nQP 36 QP 36 )      61704 bits [Y 32.6654 dB    U 42.8239 dB    V 44.2602 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ba78e7b5ac129523e77c2c6acf5a5317,f8d319e50bd0b466457bc9d1e71fd735,c24ef4b668a09c08b081ecef4e870bdd]
    POC   11 TId: 0 ( I-SLICE, nQP 36 QP 36 )      37824 bits [Y 33.4551 dB    U 41.6210 dB    V 38.4997 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4edee56e4a35d62b60197623b6959076,86d7b1ddebb5100a21898d119e4e3d51,7cb2764f38570768b5d848581d7bc910]
    POC   12 TId: 0 ( I-SLICE, nQP 36 QP 36 )     127056 bits [Y 30.6852 dB    U 38.2885 dB    V 38.7273 dB] [ET     3 ] [L0 ] [L1 ] [MD5:07854843e25607457af74021255d4ef8,5a002544f7ff3c7e750384f46a430175,54bca63f60162cbaaffa003616fee711]
    POC   13 TId: 0 ( I-SLICE, nQP 36 QP 36 )     156528 bits [Y 29.5139 dB    U 41.6671 dB    V 40.8793 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b5cda4096674194c6063566bb712af51,abd45454c4bede9251601931ba5a056b,56cb1923fae1b5ec18406416336d5f61]
    POC   14 TId: 0 ( I-SLICE, nQP 36 QP 36 )      51032 bits [Y 33.7235 dB    U 41.9117 dB    V 39.1385 dB] [ET     2 ] [L0 ] [L1 ] [MD5:be2c024d40deb2f2a74a14b626ce09f6,2ac75ab58b7f6071c760d0c1cbd2421d,91a85bd94e26dd878a980af7d4e079ec]
    POC   15 TId: 0 ( I-SLICE, nQP 36 QP 36 )     146232 bits [Y 30.9189 dB    U 38.1651 dB    V 39.6573 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3fa5f205b549be1e24662b22ecea984c,40fda8a2b21889df302c897e08b0d850,5bb4a3094e8704fee12e0e8d52b31c3e]
    POC   16 TId: 0 ( I-SLICE, nQP 36 QP 36 )     141944 bits [Y 30.6807 dB    U 38.3613 dB    V 38.9435 dB] [ET     3 ] [L0 ] [L1 ] [MD5:15ce9d35e73585e6315d9947e41cf35b,e8b3c82232b2c128186d2c4fb5d01124,f61433e828c681bd98fbd6457cdbb2d2]
    POC   17 TId: 0 ( I-SLICE, nQP 36 QP 36 )      71968 bits [Y 31.9787 dB    U 41.4314 dB    V 42.0305 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5b62efcb4797ad9667164b51d7585ba1,f4c6ef224fb0ddd5559a311cbfd57dbe,82789be9b5e688c8d57a315b2bd79166]
    POC   18 TId: 0 ( I-SLICE, nQP 36 QP 36 )      79696 bits [Y 31.6764 dB    U 39.1109 dB    V 39.5362 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ccaed61e2c9f1d89c1f74d5c655bd9ee,5408ef72d97dcff47acaf73ba826e212,26c5edf33daf03cd3af053c257bb7e54]
    POC   19 TId: 0 ( I-SLICE, nQP 36 QP 36 )      44312 bits [Y 36.0989 dB    U 40.8975 dB    V 40.5652 dB] [ET     2 ] [L0 ] [L1 ] [MD5:67a7bd9fd466ad460a7724af04bc7048,74dff0f38b3c1f53a3b61ede32b5ea16,c858ded9dcfc2c0c2938b5a674c6d296]
    POC   20 TId: 0 ( I-SLICE, nQP 36 QP 36 )      53736 bits [Y 34.4823 dB    U 42.0246 dB    V 43.3403 dB] [ET     2 ] [L0 ] [L1 ] [MD5:536d5e12ec51296b0d81a2e84abf3aec,c7710c43d0e104264c31239716cec829,5cb29d0a227bbf64384581f7ba610287]
    POC   21 TId: 0 ( I-SLICE, nQP 36 QP 36 )     112432 bits [Y 31.6953 dB    U 40.2875 dB    V 41.9801 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7bc924d9ebb23c10d3ff6fcf96f32cc8,2e2db215f66d545ddc8a985d8a46be43,3d9a59641685eba27ae21bd65e45b351]
    POC   22 TId: 0 ( I-SLICE, nQP 36 QP 36 )      50744 bits [Y 33.9299 dB    U 40.8423 dB    V 42.9969 dB] [ET     2 ] [L0 ] [L1 ] [MD5:812dd0d204f7ef62087188104e5d4b25,058eb0d9207415a5ec8bed808db54b3c,22353e9c0958ceaafa195e7783fe5d20]
    POC   23 TId: 0 ( I-SLICE, nQP 36 QP 36 )     186656 bits [Y 30.0810 dB    U 38.8448 dB    V 38.3970 dB] [ET     3 ] [L0 ] [L1 ] [MD5:196efe27586a1cd90bd4b129ea42b080,02d211e4b1c788fbfa4cf04770b78d25,09e67e0196303a89d2fd9290b924f56c]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    5751.0000   32.3747   40.5536   40.6671   33.4044  

# 35
    POC    0 TId: 0 ( I-SLICE, nQP 35 QP 35 )     105288 bits [Y 32.3563 dB    U 41.6813 dB    V 39.9452 dB] [ET     3 ] [L0 ] [L1 ] [MD5:61639a8220780fb4bcb3c503d02f76d6,87992c17fa048ef75566b33cacd1f7ec,da53edcf56c814368dbab125bf5e4a60]
    POC    1 TId: 0 ( I-SLICE, nQP 35 QP 35 )     209080 bits [Y 30.8229 dB    U 38.2352 dB    V 38.6218 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7fde6b92565f1e21b844f73015020723,0a6193de6b41314775806c0971b8ab3a,c79e2a9053adc5913f9f4e52394debe5]
    POC    2 TId: 0 ( I-SLICE, nQP 35 QP 35 )      61480 bits [Y 33.7323 dB    U 43.6194 dB    V 38.6992 dB] [ET     2 ] [L0 ] [L1 ] [MD5:25927ae9bcb19bcdcc0172374a6defb7,b90b089d967d5169f7daaedc6a9c8a98,d6bc4cd0e9292d5db56e1c007662eb2c]
    POC    3 TId: 0 ( I-SLICE, nQP 35 QP 35 )      66696 bits [Y 34.6823 dB    U 41.5144 dB    V 42.2316 dB] [ET     2 ] [L0 ] [L1 ] [MD5:53bb62549f8782691dffb52fcd2a3118,ffa4b1c04e6e110d3111038080e33d3d,7b1f3762b66115057d2b8e5220049ae4]
    POC    4 TId: 0 ( I-SLICE, nQP 35 QP 35 )     132872 bits [Y 31.7040 dB    U 41.6922 dB    V 41.9481 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ff400bfc83dc49729b076dbfcd5b11be,f18b2c3f4834ad91082ac7c92eb8cc39,e6b954ca54e5e0b0aaa470c270ac2604]
    POC    5 TId: 0 ( I-SLICE, nQP 35 QP 35 )      46992 bits [Y 34.6281 dB    U 43.0642 dB    V 42.8586 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4b1381f655b89a4f331aaba4e5e46341,157f6e4e9574586e508c4dd1a6c5a3c8,d3426ae83ead7d2099ce1f4dcc0b4e31]
    POC    6 TId: 0 ( I-SLICE, nQP 35 QP 35 )     292264 bits [Y 29.0917 dB    U 38.5945 dB    V 41.0141 dB] [ET     3 ] [L0 ] [L1 ] [MD5:43a8cb63dc238ccb20635720c37067c6,ed6629ec7778ce93c58cc531b7017aad,d14dd5e34206c1bea64f757926015928]
    POC    7 TId: 0 ( I-SLICE, nQP 35 QP 35 )      81632 bits [Y 34.9564 dB    U 39.9861 dB    V 41.2976 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5f8a1ef236575d6e6ad3f976b876bbab,3c698b9e8c346d8a4bade4b93ae04355,3331571113efd6a06e16cd360157ff6f]
    POC    8 TId: 0 ( I-SLICE, nQP 35 QP 35 )      50656 bits [Y 35.3800 dB    U 41.8257 dB    V 42.9321 dB] [ET     2 ] [L0 ] [L1 ] [MD5:767ef2fd06652af05895c9ac003ed96e,3205cf720e87c79ec5fdb9f0d62bee6a,8ddf8d67a7fe3ebda66027a6b157ecb4]
    POC    9 TId: 0 ( I-SLICE, nQP 35 QP 35 )      79160 bits [Y 33.9505 dB    U 42.5683 dB    V 43.1777 dB] [ET     2 ] [L0 ] [L1 ] [MD5:79e25766a26d244000eda2559bede3bd,5ba097a3c97707e6e95e7a2a04d01daf,021aa934203f0018146e48f84dcac779]
    POC   10 TId: 0 ( I-SLICE, nQP 35 QP 35 )      72520 bits [Y 33.2010 dB    U 43.1499 dB    V 44.5614 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5bbd45d74376a0a8d88a89961967e223,e0304f32a5fe299c9e79e8903c101c12,1a30af2bb2a6ccf662db73e47f2dd2e6]
    POC   11 TId: 0 ( I-SLICE, nQP 35 QP 35 )      46392 bits [Y 33.9272 dB    U 42.0553 dB    V 38.9228 dB] [ET     2 ] [L0 ] [L1 ] [MD5:bfd6dc0d0026a6bdc030856682b695d5,e1f9acf0149e84759f3ba93e6de8132d,69dfef3631c350373204db1a14b47f7d]
    POC   12 TId: 0 ( I-SLICE, nQP 35 QP 35 )     147760 bits [Y 31.2962 dB    U 38.7966 dB    V 39.1384 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8693a00e1434c1dbff1c4f218799ca62,299a30590dea4dbeca935fbaaee89dc4,545be1b7673b2b69254b45b3cfc55f61]
    POC   13 TId: 0 ( I-SLICE, nQP 35 QP 35 )     182808 bits [Y 30.1560 dB    U 42.1691 dB    V 41.3585 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6aba88caa5e33eac7c581cbf046619d0,0463349ca6311c77f67ec753923704b1,60026bd5c16cba3a6e4d2dbdca84ad43]
    POC   14 TId: 0 ( I-SLICE, nQP 35 QP 35 )      59760 bits [Y 34.2164 dB    U 42.3396 dB    V 39.5788 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cfab87186f5e56e1f3d408e8bf943004,b0a293d45dd25b9f78af3b46a215ec0a,645a9610e990bc9e44ba6a67dbd8e632]
    POC   15 TId: 0 ( I-SLICE, nQP 35 QP 35 )     167048 bits [Y 31.6331 dB    U 38.6273 dB    V 40.0136 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4cb48790ef4ff878f5a4aa6475d8298b,c7928d22c8960c8be4e1d91862b5bc47,665add8ee7ddacb7680dc028e6f10534]
    POC   16 TId: 0 ( I-SLICE, nQP 35 QP 35 )     165528 bits [Y 31.4064 dB    U 38.8409 dB    V 39.3207 dB] [ET     2 ] [L0 ] [L1 ] [MD5:8808a6ca80a2b4d2e9522d22aa977dcb,3415c16210fc863c830ae26eb977f11c,af26423cd6d118748e36d181f3e250cd]
    POC   17 TId: 0 ( I-SLICE, nQP 35 QP 35 )      84632 bits [Y 32.5288 dB    U 41.9630 dB    V 42.3084 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7db6ac85f3e279f3a2291a1e6455e35f,5e989aaf70b41d5eebd4657954c5ac91,61f92013aefe4bbd088c8d8b1aacc098]
    POC   18 TId: 0 ( I-SLICE, nQP 35 QP 35 )      95248 bits [Y 32.2646 dB    U 39.5429 dB    V 40.0623 dB] [ET     2 ] [L0 ] [L1 ] [MD5:604621964f0052a0807dabde93befc31,b4924d3af06db9629b36d512cd90262e,94df43f83f45f2d8ec4b009e775fcc86]
    POC   19 TId: 0 ( I-SLICE, nQP 35 QP 35 )      49952 bits [Y 36.6757 dB    U 41.5620 dB    V 41.2744 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b74e00dc524051e8a090e1668db5a0fc,4099fa09fbf3c22bb3bc641e31472702,2d2cde1b3213145dbae0064898975bb2]
    POC   20 TId: 0 ( I-SLICE, nQP 35 QP 35 )      61640 bits [Y 35.0769 dB    U 42.6369 dB    V 43.9561 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3af7305b558b64cc0685b2265c0dc490,4d3257be6654ac4b858aae31649fec6f,aa9f923af17c3f564fc6a49c07e0787f]
    POC   21 TId: 0 ( I-SLICE, nQP 35 QP 35 )     129392 bits [Y 32.3528 dB    U 40.7757 dB    V 42.3932 dB] [ET     2 ] [L0 ] [L1 ] [MD5:72c45d78e2d1bda33bf2f225ce776329,1790d6ffeab04325bc3483b430ce59e4,67f5f1b601e948b9e77be5c387846bb5]
    POC   22 TId: 0 ( I-SLICE, nQP 35 QP 35 )      58256 bits [Y 34.4304 dB    U 41.7530 dB    V 43.1372 dB] [ET     2 ] [L0 ] [L1 ] [MD5:301485b991c511fa1faff019441005e5,f36f1a19a244775622030c40f1907b3c,abda7a6217dcf0cde600535daa317565]
    POC   23 TId: 0 ( I-SLICE, nQP 35 QP 35 )     210552 bits [Y 30.7704 dB    U 39.4825 dB    V 39.0267 dB] [ET     3 ] [L0 ] [L1 ] [MD5:fce3b34b870c9870b0ab0ccfa16de33a,13d959533bbd94566bd921e168c6878e,5cd068e9de4dcde859817e040e5f5c12]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    6644.0200   32.9683   41.1032   41.1574   34.0186  

# 34
    POC    0 TId: 0 ( I-SLICE, nQP 34 QP 34 )     122328 bits [Y 32.9760 dB    U 41.8464 dB    V 39.7631 dB] [ET     3 ] [L0 ] [L1 ] [MD5:335e4de9bf36b9dff9bf26434370b842,69ba450e9cb57be108673e5ea2986231,1397690692b955e37be30c834b37b7bd]
    POC    1 TId: 0 ( I-SLICE, nQP 34 QP 34 )     235936 bits [Y 31.6050 dB    U 38.1408 dB    V 38.5736 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6eccb78a7c70214b1ec07ff177fe2e6d,ac75bc8655942cecc367605b07c724c6,d7b1ae55ef560cb887e1371b70f52dec]
    POC    2 TId: 0 ( I-SLICE, nQP 34 QP 34 )      72520 bits [Y 34.2734 dB    U 43.7526 dB    V 38.6530 dB] [ET     2 ] [L0 ] [L1 ] [MD5:30035ffbf77a937aa477b0c77fe65721,f1ac44b492c517f4b75fa2d27bc81d8c,68d9c38dcc706261b9492e7e6ec1c96b]
    POC    3 TId: 0 ( I-SLICE, nQP 34 QP 34 )      74760 bits [Y 35.2688 dB    U 41.4376 dB    V 42.1465 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7aef830a9e3e79bedde585787e041828,de9ae45818573b2f8dbe01ae75cce256,f84b486b2dde630111e2e0b8541a19d7]
    POC    4 TId: 0 ( I-SLICE, nQP 34 QP 34 )     153416 bits [Y 32.4327 dB    U 41.5907 dB    V 42.0001 dB] [ET     3 ] [L0 ] [L1 ] [MD5:80be8f7ae7e876b357fadb28faed0830,8a5ae5a626c4a014bda8ac7a22ae6fca,5728cf146b908d587fb77361266892df]
    POC    5 TId: 0 ( I-SLICE, nQP 34 QP 34 )      54824 bits [Y 35.1681 dB    U 43.1324 dB    V 42.8250 dB] [ET     2 ] [L0 ] [L1 ] [MD5:cec91690554e9af16581b94430ea6b39,5bfb1f3ec552849e05d332c26cd69d37,d7ecf6874675b60796b063f5c0505a3e]
    POC    6 TId: 0 ( I-SLICE, nQP 34 QP 34 )     334952 bits [Y 29.8687 dB    U 38.5652 dB    V 40.9990 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8714d63e3347f07d01abf1a78bad6bdb,42b5bbcce5b4e3066dc4edb2e420ef9f,aaddd8cdebe26c08b307e32f39aef690]
    POC    7 TId: 0 ( I-SLICE, nQP 34 QP 34 )      90752 bits [Y 35.6457 dB    U 39.9611 dB    V 41.0860 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9c42e53086168b8f6efcc2cee77e51bb,99518bea5b3e58ba17f91d151e31ce3d,e491fa7c1287b9f2da4f2cfc5908c74c]
    POC    8 TId: 0 ( I-SLICE, nQP 34 QP 34 )      58288 bits [Y 35.9829 dB    U 41.7214 dB    V 42.8586 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5a7a88afeb81f75fa5b5b18493be6998,9439ea52d8a959c5ab6b2b91236b202e,79c8c533e0bac6d8b135d46e15331b20]
    POC    9 TId: 0 ( I-SLICE, nQP 34 QP 34 )      89784 bits [Y 34.5737 dB    U 42.5082 dB    V 43.1293 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6d7318f312989d5706e373fcc827de3b,96b2267f3c6f4203f53163b9cd2622c2,74e7547cb72431faa88897ef61e9e4e9]
    POC   10 TId: 0 ( I-SLICE, nQP 34 QP 34 )      85568 bits [Y 33.8032 dB    U 42.9403 dB    V 44.4970 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3765977daba233262cf2e980f90eb838,00efb6a5478555360a6616e8aad9e520,d225c0fb52bd2038f04667b431e85a72]
    POC   11 TId: 0 ( I-SLICE, nQP 34 QP 34 )      54960 bits [Y 34.3714 dB    U 41.8697 dB    V 38.9141 dB] [ET     2 ] [L0 ] [L1 ] [MD5:081f7e04edce86c843ace51f4d085993,0126070f7e786b41ee05f78b29b6fd33,624f0918e9331324767d727555a362f0]
    POC   12 TId: 0 ( I-SLICE, nQP 34 QP 34 )     169240 bits [Y 31.9390 dB    U 38.7960 dB    V 39.3010 dB] [ET     3 ] [L0 ] [L1 ] [MD5:cf80aacff5be171bd438ce6dd19c6c3f,371ea91b3f8270b9b2585c8121758df9,34c59efaf397e4c7ff1c9cd642be6b60]
    POC   13 TId: 0 ( I-SLICE, nQP 34 QP 34 )     213864 bits [Y 30.8817 dB    U 41.9103 dB    V 41.1615 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e9847242a5c1b3dba628824bacf62216,e3312049b80ff6fab0da62034643eec9,6b97dfc9324a19b0f361edb59155d3e8]
    POC   14 TId: 0 ( I-SLICE, nQP 34 QP 34 )      69648 bits [Y 34.8089 dB    U 42.2946 dB    V 39.4861 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e22da6f3d143a9441781d9e1f5e575b8,e536d5cf604af1145faab0557bb99a99,c3ee819e0d5a98d87146ae1a12f13e0f]
    POC   15 TId: 0 ( I-SLICE, nQP 34 QP 34 )     189936 bits [Y 32.3949 dB    U 38.6110 dB    V 39.9601 dB] [ET     3 ] [L0 ] [L1 ] [MD5:bd2fb81622f6ff40f1c054f4e5ed2b6f,8f045725ace683679477936c08f88c98,50fba53f8fc072d1604977bfbce6ea80]
    POC   16 TId: 0 ( I-SLICE, nQP 34 QP 34 )     187952 bits [Y 32.0818 dB    U 38.8356 dB    V 39.2929 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c1478a6a5504072288080d5b9633aecc,a2a7821eb5744460931a4787861d3a28,c7c76b86c301058e4ee7179da243665f]
    POC   17 TId: 0 ( I-SLICE, nQP 34 QP 34 )      99456 bits [Y 33.1149 dB    U 42.0075 dB    V 42.3303 dB] [ET     2 ] [L0 ] [L1 ] [MD5:f31dfdb29aaa5fd0d2ba976bd50aa8d1,f499303189cf9c23a1127c655a5bb2af,44c76fbb472fce76e02be7c44ac77ea1]
    POC   18 TId: 0 ( I-SLICE, nQP 34 QP 34 )     112744 bits [Y 32.9128 dB    U 39.5197 dB    V 39.9580 dB] [ET     2 ] [L0 ] [L1 ] [MD5:af9cb805809e38657adc57ea95d58b38,6ca310c809a8d45c0410c6c210a3b016,1eb47aa98506adb7e7522f3f9fa18478]
    POC   19 TId: 0 ( I-SLICE, nQP 34 QP 34 )      54712 bits [Y 37.1414 dB    U 41.4327 dB    V 41.0840 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3b88371b51784124c986880ea279ed87,0091f808c8887aca0ddc3b2e23ff9797,3c6d26264b23ec3bf0b1556832d41334]
    POC   20 TId: 0 ( I-SLICE, nQP 34 QP 34 )      69552 bits [Y 35.7089 dB    U 42.3904 dB    V 43.7802 dB] [ET     2 ] [L0 ] [L1 ] [MD5:62874f6f567208a8747e7b8395b510d5,d319c45e3d5cbe76e02cf13f0502bd09,5f2f743fa7dcd16d7857fcb95cecb65a]
    POC   21 TId: 0 ( I-SLICE, nQP 34 QP 34 )     147920 bits [Y 33.0549 dB    U 40.7707 dB    V 42.4170 dB] [ET     2 ] [L0 ] [L1 ] [MD5:537cff85b7a20a8c81e1ba84add50bb0,d1a6eff8afb8fa878a46899f574b05e0,a7d6f9cc9faaf5b60a8fb4c5019dcfe8]
    POC   22 TId: 0 ( I-SLICE, nQP 34 QP 34 )      68352 bits [Y 35.0648 dB    U 41.7200 dB    V 43.2258 dB] [ET     2 ] [L0 ] [L1 ] [MD5:732280adf3fb277a5a29c8c695d57512,980ea4fa6761400663dd655138ac4202,c3395f1bb5236f90851cfa4abd45c7e2]
    POC   23 TId: 0 ( I-SLICE, nQP 34 QP 34 )     236056 bits [Y 31.5085 dB    U 39.3619 dB    V 38.9745 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d3a1778f40daf10480549dedfab17f76,3d8ee51468b413a7b00da4d93b68c639,31e60784b2e11953b5e2a15388bc6f4f]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    7618.8000   33.6076   41.0465   41.1007   34.6310  

# 33
    POC    0 TId: 0 ( I-SLICE, nQP 33 QP 33 )     140952 bits [Y 33.6075 dB    U 42.3539 dB    V 40.2672 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d55a9be2245e84cb055ad205c8867c97,1764b8bf2a85beac875f40af2a6a1b84,3e508ed8b7c7ca558fc725324bda0f77]
    POC    1 TId: 0 ( I-SLICE, nQP 33 QP 33 )     268504 bits [Y 32.4010 dB    U 38.7341 dB    V 39.0773 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d166c1bff04410a3806c4921b03441e3,6a4c88710e79270d52b70eb0654c2258,1a8c0d7f0a0d44b1a8d5b9c0312570e9]
    POC    2 TId: 0 ( I-SLICE, nQP 33 QP 33 )      84152 bits [Y 34.7834 dB    U 43.7486 dB    V 39.1851 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e7d83852d4bcb69146955ba331141ef1,06ab7f797d83bf6101a36768c0766142,95ffbc8d67642e78855dac667d0535b5]
    POC    3 TId: 0 ( I-SLICE, nQP 33 QP 33 )      85248 bits [Y 35.9038 dB    U 42.0257 dB    V 42.8706 dB] [ET     3 ] [L0 ] [L1 ] [MD5:389184860ac2d8f59c68ca64d4be2a1f,f23b74e3de76634312913459ece49071,ccd8ea55efb48710e1b7dd5e569c1e0a]
    POC    4 TId: 0 ( I-SLICE, nQP 33 QP 33 )     176456 bits [Y 33.1028 dB    U 41.9363 dB    V 42.4555 dB] [ET     3 ] [L0 ] [L1 ] [MD5:503c5bf539246dc36cc2abb937cc9095,cbf056ff7ec752b6d8875a50c2973323,3cc3be018a77de9b7f18d200e8c61d6f]
    POC    5 TId: 0 ( I-SLICE, nQP 33 QP 33 )      63968 bits [Y 35.7034 dB    U 43.9838 dB    V 43.1731 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e94659e28b8f32a5eb3d3c2993abbec3,4e3e747ef5afa7de02f1c749e98a29c5,35037867aeaa5c51515981950e9a081a]
    POC    6 TId: 0 ( I-SLICE, nQP 33 QP 33 )     385248 bits [Y 30.7224 dB    U 39.0463 dB    V 41.2533 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6a6b330fb327a316de94d23d82fa1b07,4a6f43a452a13285a6d89a439b005175,62793abeaf9e77782493c19e38a964a4]
    POC    7 TId: 0 ( I-SLICE, nQP 33 QP 33 )     102272 bits [Y 36.3295 dB    U 40.6734 dB    V 41.6532 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ea9280cdf909ab7956d49f429c41917e,a530e3ed7d83097fc0bd5780da870c08,46741d750a6573f155e34c52f68b52ee]
    POC    8 TId: 0 ( I-SLICE, nQP 33 QP 33 )      67768 bits [Y 36.6382 dB    U 42.4431 dB    V 43.2801 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d0f0c91bdcba62dac40c8a5b23a16522,20d5d4d07d8083c715149666664e3a08,eb5b5b1dc4ddf1f9d55d8717bb0e58a2]
    POC    9 TId: 0 ( I-SLICE, nQP 33 QP 33 )     102240 bits [Y 35.1819 dB    U 42.8273 dB    V 43.6525 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c75dc95e386c931eb38f29d2be8ea9f6,4f67c9336ed7ea745c93c30817c3e5f3,da57c6642a152265be7bd674c218a005]
    POC   10 TId: 0 ( I-SLICE, nQP 33 QP 33 )     100720 bits [Y 34.3914 dB    U 43.4575 dB    V 45.0053 dB] [ET     2 ] [L0 ] [L1 ] [MD5:6479e10c599d71ee4d9c274b3a75c635,f98d635ac90da8f793a2817bd21ff466,f630fa68ba267bd5e8a9091c7bb608e6]
    POC   11 TId: 0 ( I-SLICE, nQP 33 QP 33 )      67552 bits [Y 34.8833 dB    U 42.3799 dB    V 39.4378 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c385e301422e154e14422b1184ba51f5,c3e1e04c6510b5f256348d3e738a9930,0c0245af8550159697c750a031b0cd42]
    POC   12 TId: 0 ( I-SLICE, nQP 33 QP 33 )     196584 bits [Y 32.6312 dB    U 39.3793 dB    V 39.7916 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e0651654da2c4f39114d0c6ff4f4aa4f,e287e5aded765035bb64ab0e7cd2fbb3,56d764d179bae8d227665fe8d40cb84d]
    POC   13 TId: 0 ( I-SLICE, nQP 33 QP 33 )     248336 bits [Y 31.6350 dB    U 42.4222 dB    V 41.7377 dB] [ET     3 ] [L0 ] [L1 ] [MD5:dc38d9560c2695778645132b6deb6f40,67c201e754afe024832c9a62a7ea39ff,7bed995db2230313009ecdb34246941d]
    POC   14 TId: 0 ( I-SLICE, nQP 33 QP 33 )      81384 bits [Y 35.3455 dB    U 42.9861 dB    V 40.1366 dB] [ET     2 ] [L0 ] [L1 ] [MD5:832edef10ea6c7330756d9570dcd62ac,7bc610f15bf01df0db59182978aacefa,3e739a6ca4af062f85550fe3c11c7ac9]
    POC   15 TId: 0 ( I-SLICE, nQP 33 QP 33 )     215448 bits [Y 33.1336 dB    U 39.0688 dB    V 40.3187 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f77ea5bbb56459ac1c84d70db11b8ba7,33cba896a78eafd6e6871f6f966e3628,22e1737cf21a1927cc2c1bd519949b06]
    POC   16 TId: 0 ( I-SLICE, nQP 33 QP 33 )     214624 bits [Y 32.8034 dB    U 39.2481 dB    V 39.6850 dB] [ET     3 ] [L0 ] [L1 ] [MD5:dc12818d1355c61566e53b2e71769e76,70bfb5ed507d6dded91a4d846e4c0226,12fa3e35e3a65970c5cdc0b12db8cd17]
    POC   17 TId: 0 ( I-SLICE, nQP 33 QP 33 )     117848 bits [Y 33.7739 dB    U 42.2436 dB    V 42.8408 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a4e260924fdac3322534d0be13435dec,92687537fd52a72180d4397475719390,669e7ab2a275afedce9a4dfbe8dfcc23]
    POC   18 TId: 0 ( I-SLICE, nQP 33 QP 33 )     131824 bits [Y 33.5001 dB    U 39.9994 dB    V 40.4641 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ca201e7a582ace1c431471617df34672,7163ada4323bbaad604fcd6b74339ace,422ad0cf61149c5a9ebbf7ffe0fd49ae]
    POC   19 TId: 0 ( I-SLICE, nQP 33 QP 33 )      62688 bits [Y 37.7385 dB    U 42.1692 dB    V 41.6773 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2cbcc37f10c3d02262258fc171bdc438,49b8ff8a344a8c85f749bbfb41688b41,21e7ffb8e4e9ff43bb6a5543e3bc3701]
    POC   20 TId: 0 ( I-SLICE, nQP 33 QP 33 )      78304 bits [Y 36.3077 dB    U 42.8702 dB    V 44.3920 dB] [ET     2 ] [L0 ] [L1 ] [MD5:b1cae3197f8a8094c1276999dff9fdc4,12534dadcde75d403f4b88013a4a9243,93348a8dd156d740a382163548985bfa]
    POC   21 TId: 0 ( I-SLICE, nQP 33 QP 33 )     168896 bits [Y 33.7962 dB    U 41.2523 dB    V 42.8102 dB] [ET     2 ] [L0 ] [L1 ] [MD5:e4757f426dd194cb42ee493a7d115dac,224ed1cb8bad4774070f704b9d291365,9ad3ed9671788c657294debf8321f517]
    POC   22 TId: 0 ( I-SLICE, nQP 33 QP 33 )      79520 bits [Y 35.6987 dB    U 42.1312 dB    V 43.7364 dB] [ET     2 ] [L0 ] [L1 ] [MD5:de0625a840eb34d2e9703d31a4dcbeff,c8a16d735967cf88149596356dc0f4ba,082cf6eb1c66f3d329962429f913db33]
    POC   23 TId: 0 ( I-SLICE, nQP 33 QP 33 )     265392 bits [Y 32.2387 dB    U 39.9845 dB    V 39.5731 dB] [ET     3 ] [L0 ] [L1 ] [MD5:74882226a28b0da53fe4e9d1e985ce0e,7154b05d3fff326ec4841df56a213b07,5ebadde615f56d3ccc2fc7eefd66f77d]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a    8764.8200   34.2605   41.5569   41.6031   35.2972  

# 32
    POC    0 TId: 0 ( I-SLICE, nQP 32 QP 32 )     163336 bits [Y 34.3201 dB    U 42.8465 dB    V 40.9071 dB] [ET     3 ] [L0 ] [L1 ] [MD5:58b6406401ce072470f3d46c326f1b71,dfa9303fec279f0a801e3218dbdb921d,70c8640ccfad7154e742063ffad6556e]
    POC    1 TId: 0 ( I-SLICE, nQP 32 QP 32 )     299976 bits [Y 33.1659 dB    U 39.2965 dB    V 39.7224 dB] [ET     3 ] [L0 ] [L1 ] [MD5:43ec8ca4ede158a7f7101979d107c5e7,dbcaf4e22632da069aabcdef307b37d2,294db736cc7ae0059115dfeb7b4a675b]
    POC    2 TId: 0 ( I-SLICE, nQP 32 QP 32 )      99560 bits [Y 35.3526 dB    U 44.5197 dB    V 39.7953 dB] [ET     3 ] [L0 ] [L1 ] [MD5:72d07cdce4ea5a3bdaab759cbc0cda89,de12f493eb932a93f4ee16f95f9172a1,e34878b6fc3bb80209b681d72a9c118d]
    POC    3 TId: 0 ( I-SLICE, nQP 32 QP 32 )      97240 bits [Y 36.5381 dB    U 42.5839 dB    V 43.5648 dB] [ET     2 ] [L0 ] [L1 ] [MD5:55e5d0eab73d090e61ffc026608516da,6647fa7d059f1011c18a99d60c45362d,ad57afa571811e515452e57f7d3f9745]
    POC    4 TId: 0 ( I-SLICE, nQP 32 QP 32 )     202320 bits [Y 33.8302 dB    U 42.6213 dB    V 42.8413 dB] [ET     3 ] [L0 ] [L1 ] [MD5:dd416204c6cd9eff577577b96ae1fa04,edbc3fdd68b76daf1f35af4fd07d905d,95816b5096d84c71461cc2432bdedcff]
    POC    5 TId: 0 ( I-SLICE, nQP 32 QP 32 )      75248 bits [Y 36.2710 dB    U 44.5462 dB    V 43.8471 dB] [ET     2 ] [L0 ] [L1 ] [MD5:20b1d5e8ebc595bf6ad42f585c7c3eda,643167c2ac687338b35114219a75f48d,c32bee13c4db582c32935fd7b523c251]
    POC    6 TId: 0 ( I-SLICE, nQP 32 QP 32 )     437592 bits [Y 31.5886 dB    U 39.4718 dB    V 41.8183 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d98051be6f373247bb30d70f5bfa946e,4a1fe377b3bbaa81646cd359af9c7b4f,b9c1a6b874d69aaba2edda42ecae0f6c]
    POC    7 TId: 0 ( I-SLICE, nQP 32 QP 32 )     113408 bits [Y 36.9658 dB    U 41.2549 dB    V 42.0105 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7ff7f3867b8884148941c436f07ae4dc,40e4f609bc024b62208f9c4827619006,5eef812b0eabf39fd6f8ae7afa6ed902]
    POC    8 TId: 0 ( I-SLICE, nQP 32 QP 32 )      77680 bits [Y 37.2838 dB    U 43.0981 dB    V 43.9141 dB] [ET     3 ] [L0 ] [L1 ] [MD5:30b27ed964e3ebfde2f7d729f5763308,e2ca45a38e2e90f016113b0fccd55f32,a44882685647ee3350efda1050edc130]
    POC    9 TId: 0 ( I-SLICE, nQP 32 QP 32 )     115872 bits [Y 35.8004 dB    U 43.3313 dB    V 43.9750 dB] [ET     3 ] [L0 ] [L1 ] [MD5:97ec88139c9f6ab427eda64ba99a3b5a,7da7d16e21c380293796813ca2a4d178,7d35ffe356b0a9f0d9b62750f72acce8]
    POC   10 TId: 0 ( I-SLICE, nQP 32 QP 32 )     117528 bits [Y 35.0377 dB    U 43.9258 dB    V 45.4337 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e71a08d27c8431f623aab29712ea1a94,0f80c1e102edc162b0b8fcaa4b8d8a23,70a7f95e429aeb9827cb77a3a26438d6]
    POC   11 TId: 0 ( I-SLICE, nQP 32 QP 32 )      80544 bits [Y 35.4114 dB    U 42.8223 dB    V 39.8017 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ae429a3caf4aeff6c763db0727e4b88d,b63973da47a7f8507b5b017db174a555,04ef5c3ab9ea2c8a4586f2e8f783cc33]
    POC   12 TId: 0 ( I-SLICE, nQP 32 QP 32 )     224800 bits [Y 33.3186 dB    U 39.9228 dB    V 40.3770 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c38d37e20ecf3c167cb57fb2fadae4d6,9ade9d0e1a722a6c98f35c77feb05fde,14dd410c67ef4b4c6da2a0f5a2850261]
    POC   13 TId: 0 ( I-SLICE, nQP 32 QP 32 )     285864 bits [Y 32.4201 dB    U 42.7227 dB    V 42.1763 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8373691d2345723e82dd7e7f45e854d7,510455ccb8440aabb27d646a9f2aab31,7510fa6de66e62cd8121d4754800c853]
    POC   14 TId: 0 ( I-SLICE, nQP 32 QP 32 )      95608 bits [Y 35.9512 dB    U 43.4746 dB    V 40.7264 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a75d78d8f76dc6c131bdf185d238d775,7fc2610caf5cbe2256e787ee823a766b,3f39dc9ff32db9467f2b32cd26b77bf1]
    POC   15 TId: 0 ( I-SLICE, nQP 32 QP 32 )     244368 bits [Y 33.9325 dB    U 39.5917 dB    V 40.9519 dB] [ET     4 ] [L0 ] [L1 ] [MD5:c083752f8dee6299e20164b06cde2397,0b10002b80d0eacf014a5a3a447d55c8,56e96f3f6da835d783f80e2e6b76399c]
    POC   16 TId: 0 ( I-SLICE, nQP 32 QP 32 )     244144 bits [Y 33.5401 dB    U 39.7643 dB    V 40.1869 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f1ec4a249c996c26d5c91fb59b6c5a04,af2d100422e732225e9af493ad7ebb18,18f0a548ed3a9e34badc9211fbb90c6f]
    POC   17 TId: 0 ( I-SLICE, nQP 32 QP 32 )     136480 bits [Y 34.4048 dB    U 42.6872 dB    V 43.3505 dB] [ET     3 ] [L0 ] [L1 ] [MD5:fda37236a585c303c319f9574bc714b6,e5f75977bb2a23ccdbcb3b4b52d492a5,bbe0cc160184c138e35b83390d442520]
    POC   18 TId: 0 ( I-SLICE, nQP 32 QP 32 )     154064 bits [Y 34.1888 dB    U 40.3642 dB    V 40.9155 dB] [ET     2 ] [L0 ] [L1 ] [MD5:790c7094a8b88745f94ad9283e678fc6,c8fc8852a2d7c21e59e855f4bdd43289,b01bd5e505ef8e879a5c0e16c59410c7]
    POC   19 TId: 0 ( I-SLICE, nQP 32 QP 32 )      70560 bits [Y 38.2621 dB    U 42.6483 dB    V 42.2718 dB] [ET     2 ] [L0 ] [L1 ] [MD5:7bad219e49ee992fe35110dc6d1733f2,2cfcbe7266eafbf8235d2ef74df04fcc,675a76427caa29de022b59813b56620b]
    POC   20 TId: 0 ( I-SLICE, nQP 32 QP 32 )      88632 bits [Y 36.8876 dB    U 43.5180 dB    V 44.5404 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9461947f2fe8afc2bd4fb979a0de1594,28f0414ca2e12cc3ef30e8a30f2fe064,552beed04158cff73595d73e83a1fe3e]
    POC   21 TId: 0 ( I-SLICE, nQP 32 QP 32 )     191368 bits [Y 34.5353 dB    U 41.6104 dB    V 43.4432 dB] [ET     2 ] [L0 ] [L1 ] [MD5:96a677bcc52453d5bf5b1b7ca8aa40c6,d0e934d3351ea2b9d5bb4e00ce14a734,f4e64471aa0aeae562121f00fe06d111]
    POC   22 TId: 0 ( I-SLICE, nQP 32 QP 32 )      91640 bits [Y 36.3295 dB    U 42.5665 dB    V 44.1188 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d6fc68264fbb276c9e261a9ef2871363,cfb2b787ef88f965f2d4e1d04136a9e4,0c3a01d30f161b90a927fac2107a5cf0]
    POC   23 TId: 0 ( I-SLICE, nQP 32 QP 32 )     295568 bits [Y 32.9476 dB    U 40.5313 dB    V 40.1483 dB] [ET     3 ] [L0 ] [L1 ] [MD5:79dbec81ec749a64e7d48d796891acef,eede30fb956aa0101ea27ff73f3de43b,864e3b8fdc8867b7ce4b1e704c04f1a5]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   10008.5000   34.9285   42.0717   42.1183   35.9786  

# 31
    POC    0 TId: 0 ( I-SLICE, nQP 31 QP 31 )     186952 bits [Y 35.0112 dB    U 43.4021 dB    V 41.3982 dB] [ET     3 ] [L0 ] [L1 ] [MD5:5fcf573f1084e894236611a969949f12,4cf55be5752084d116631c74f2263ab0,b6528c6aa0da23da6168f9a89ef45436]
    POC    1 TId: 0 ( I-SLICE, nQP 31 QP 31 )     336512 bits [Y 33.9832 dB    U 39.9226 dB    V 40.3639 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d7cb061381b1076cb3cfe4e1fcbc02c7,0d6b1221bbc25385a462445f2079c10a,3e784924bc795932ed9958dcdbe014c6]
    POC    2 TId: 0 ( I-SLICE, nQP 31 QP 31 )     117864 bits [Y 35.9974 dB    U 45.1668 dB    V 40.5056 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b56df8b310c45477aa6eac404ccd0c1f,56c0e1ab5bec999945c51e263ae4731d,0d5845c965b6917ee06284b71e1bc158]
    POC    3 TId: 0 ( I-SLICE, nQP 31 QP 31 )     108728 bits [Y 37.1191 dB    U 42.9873 dB    V 43.8661 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ea5caa241c2c35256afc2167f6e2ca79,33ff7d0a38366f9fc7bea30d6e8b9f1e,8d219e8cea26f46445ea82555a24a67d]
    POC    4 TId: 0 ( I-SLICE, nQP 31 QP 31 )     231648 bits [Y 34.6161 dB    U 42.9168 dB    V 43.2943 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6510b287aecaa9f0b5f365dbb93d0f14,2e0273c218392499a9f62ca3d8a21e7f,16d0ebcd82c9e57c46448b51dbeb43cd]
    POC    5 TId: 0 ( I-SLICE, nQP 31 QP 31 )      88464 bits [Y 36.8316 dB    U 45.2671 dB    V 44.2725 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d6fd6152534e4df7a722fc824aed0f9b,8cfb6db0fe542abfa81ad622f8342ff7,410f91d3e6cdbacf5ee17279a0d0f684]
    POC    6 TId: 0 ( I-SLICE, nQP 31 QP 31 )     495976 bits [Y 32.4961 dB    U 39.9639 dB    V 42.2476 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e1c6b66e247fdbfeaaf347996b57af31,c9573747f24de526736d913c17b09950,25621f84c0d357439a18fb53e341e03f]
    POC    7 TId: 0 ( I-SLICE, nQP 31 QP 31 )     127424 bits [Y 37.6688 dB    U 41.9189 dB    V 42.7850 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7d8af408876501d8557105497ab758b0,a1237ac8f9f42ff1e877ea223b6087a3,bf737feef3661873ad7450a13012fa68]
    POC    8 TId: 0 ( I-SLICE, nQP 31 QP 31 )      89064 bits [Y 37.9392 dB    U 43.7614 dB    V 44.2934 dB] [ET     2 ] [L0 ] [L1 ] [MD5:dbc9b770e51b285fbf2d91602585f318,98ee50a693ac2d63383ac352e5732196,fa8c971c167315d683abd0d0d1b7a505]
    POC    9 TId: 0 ( I-SLICE, nQP 31 QP 31 )     131784 bits [Y 36.4414 dB    U 43.8295 dB    V 44.3598 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0c47eabb99d1e4aac6793d024294764f,05276af67959b55abbac5d72bf28b563,6e79a4498f691c5812ba9735fbec6a47]
    POC   10 TId: 0 ( I-SLICE, nQP 31 QP 31 )     137472 bits [Y 35.7279 dB    U 44.2302 dB    V 45.8757 dB] [ET     3 ] [L0 ] [L1 ] [MD5:cde044597a7f3d22b9a922da0552714a,efbf435c2fed68a54b0699440af86527,8ec3e9550a7b4cb572d8f08bb3ca83ce]
    POC   11 TId: 0 ( I-SLICE, nQP 31 QP 31 )      97552 bits [Y 36.0133 dB    U 43.2778 dB    V 40.2834 dB] [ET     2 ] [L0 ] [L1 ] [MD5:833b09c76f4d419e39854fdb577a6e81,6fef6865b744c5af65a77bfd79dc0922,5c131269c8c80998d2c6a8378569e585]
    POC   12 TId: 0 ( I-SLICE, nQP 31 QP 31 )     259224 bits [Y 34.0626 dB    U 40.5961 dB    V 41.0248 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4c776049ca9996319ffaac03bdf1b3c7,0ff0265a2374dde5968756da340ba66c,3b7ff7adee11832c3041da9edd9b1fdd]
    POC   13 TId: 0 ( I-SLICE, nQP 31 QP 31 )     328928 bits [Y 33.2582 dB    U 43.4742 dB    V 42.4587 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f3c6668918730db8177ad29469d7c9d2,175f1c85fdf58edecaad744416e8139f,6ea1a52f9266c7901810229f97484819]
    POC   14 TId: 0 ( I-SLICE, nQP 31 QP 31 )     111576 bits [Y 36.5644 dB    U 43.7467 dB    V 41.1951 dB] [ET     2 ] [L0 ] [L1 ] [MD5:76d904e319cd88b73ee3221e430f19bc,7098a5473fbea32ed6e2e52ea67ba349,72b11b82b2a82ce9af4834f17105240b]
    POC   15 TId: 0 ( I-SLICE, nQP 31 QP 31 )     275072 bits [Y 34.7404 dB    U 40.1295 dB    V 41.5716 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e1a500f259b3a52e7dd60d708870def1,4c25643ff0a7feff54ddc26a06a4dfda,c0c66ad46e7c8cc88860d87a032bc4d4]
    POC   16 TId: 0 ( I-SLICE, nQP 31 QP 31 )     277552 bits [Y 34.3269 dB    U 40.1987 dB    V 40.6935 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5c4b252dfd83a2b06ee44e863c3f1e44,874b94a417cd3d7c1c104a6f437bf7d7,89d58c0e54328bf8e7dd7e75610acf18]
    POC   17 TId: 0 ( I-SLICE, nQP 31 QP 31 )     159512 bits [Y 35.0956 dB    U 43.1295 dB    V 43.8140 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9652c64a77782ec304ac3fc677abf345,a010bf42efc83e70247fc3c3c7ac8a42,e9fe791f717946e83b944fefa4978170]
    POC   18 TId: 0 ( I-SLICE, nQP 31 QP 31 )     181240 bits [Y 34.9089 dB    U 40.8943 dB    V 41.4119 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1d16287faa0e945fc55d91671dc28d84,02d1a18268cd1c57b61902f3bd42405e,79fa2d401101a9eb6b9b662d310c1b52]
    POC   19 TId: 0 ( I-SLICE, nQP 31 QP 31 )      79944 bits [Y 38.8545 dB    U 43.1126 dB    V 42.7661 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9aae81749c6d4ba2ca2dcbe6b9cd5510,183328e46139b51e4e9c9e9b614ac225,95a58149e29b5eb68825ac0a0b76249c]
    POC   20 TId: 0 ( I-SLICE, nQP 31 QP 31 )     100560 bits [Y 37.5286 dB    U 43.9887 dB    V 45.1219 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3128840ec081f2625524363a37b20158,e4d134c5bf04ab5df751cebd66b67a94,9970326ffdee8ef46b7569d58a466beb]
    POC   21 TId: 0 ( I-SLICE, nQP 31 QP 31 )     217760 bits [Y 35.3320 dB    U 42.2143 dB    V 43.8697 dB] [ET     3 ] [L0 ] [L1 ] [MD5:343ec66f57623349903bf2aa4550ff47,179000cd16aa1be468961b0603dbc159,28fef1ab38bd15ba498e44be1d2c03b7]
    POC   22 TId: 0 ( I-SLICE, nQP 31 QP 31 )     106496 bits [Y 37.0560 dB    U 43.3093 dB    V 44.6013 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c318d1625d178bdb00c8cfa9b2644a39,571f7c9e44fa55aeb3e430d05f418eb6,1a0f13153cb0c1f50bba464ec0f49720]
    POC   23 TId: 0 ( I-SLICE, nQP 31 QP 31 )     331080 bits [Y 33.7163 dB    U 41.0438 dB    V 40.5976 dB] [ET     3 ] [L0 ] [L1 ] [MD5:77948c2efbce84b9e9796a1587669021,e8fac451855f49c4b6e3fc4499e6d418,18139358c8894380b08bed50f107513e]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   11445.9600   35.6371   42.6034   42.6113   36.6963  

# 30
    POC    0 TId: 0 ( I-SLICE, nQP 30 QP 30 )     213936 bits [Y 35.7154 dB    U 43.8818 dB    V 41.9464 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0795a8ae31b9fb018ee14cddb177fa0e,9f656e500886b70a4151debef49e8048,f864427db6000d6215e047380228deae]
    POC    1 TId: 0 ( I-SLICE, nQP 30 QP 30 )     375800 bits [Y 34.8118 dB    U 40.6144 dB    V 41.0841 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9bf96c1cadce57ef5cc1c802c7e140c2,735bc04b5e950b32264e17d4a9356359,e3eafdf78a0f438c001a3db2c570c9c8]
    POC    2 TId: 0 ( I-SLICE, nQP 30 QP 30 )     137480 bits [Y 36.5896 dB    U 45.7325 dB    V 40.9891 dB] [ET     3 ] [L0 ] [L1 ] [MD5:02efbbf1b58a8449c766d37b247ebc64,6f063ac9baf084039aad4a641b2fc5dc,36448d8dca46d758887b7f5753499f3a]
    POC    3 TId: 0 ( I-SLICE, nQP 30 QP 30 )     122080 bits [Y 37.6935 dB    U 43.6023 dB    V 44.4662 dB] [ET     3 ] [L0 ] [L1 ] [MD5:fbd9a361d24e15e6e33edb31f6ad951a,350a4f794d85f11c28935c51796fd217,d7547dbf8303703f75fd12ce26ade1f4]
    POC    4 TId: 0 ( I-SLICE, nQP 30 QP 30 )     263448 bits [Y 35.3968 dB    U 43.2423 dB    V 43.7473 dB] [ET     3 ] [L0 ] [L1 ] [MD5:047e3c1e1f4db01d4e6bb365c752e6ba,e1efdade540097388962cee92e4ede4a,5aba2153810d9db27a8cb780b0bd6388]
    POC    5 TId: 0 ( I-SLICE, nQP 30 QP 30 )     102552 bits [Y 37.3966 dB    U 45.7115 dB    V 44.9184 dB] [ET     3 ] [L0 ] [L1 ] [MD5:72b3d9d2f21bd2905f6ab6e9e2481a0e,6d62da956086ba61b34c559b63cabf02,297669b988059eba4cca05b4bc922130]
    POC    6 TId: 0 ( I-SLICE, nQP 30 QP 30 )     556864 bits [Y 33.4140 dB    U 40.3678 dB    V 42.7006 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e2e72937bd8f57a1ea39784b102d62ee,26da9d9b7695cad10e6cdc4622984ab7,3271fbe9951e865eec2773d7079630b8]
    POC    7 TId: 0 ( I-SLICE, nQP 30 QP 30 )     143024 bits [Y 38.4241 dB    U 42.4778 dB    V 43.3046 dB] [ET     3 ] [L0 ] [L1 ] [MD5:cd0a14eec9b6b69278533eb448b688e4,6b9b78a8364dc798042375afb2155187,1270d34acdecd4fb5c3cf624d089ba38]
    POC    8 TId: 0 ( I-SLICE, nQP 30 QP 30 )     102048 bits [Y 38.5769 dB    U 44.1425 dB    V 45.0450 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1a011c417b78a6f368a2c1d501e470e8,df7326fa566a3cdfda68e60ae7683610,a705f9f83f6835fa9028524d65b38623]
    POC    9 TId: 0 ( I-SLICE, nQP 30 QP 30 )     149736 bits [Y 37.0698 dB    U 44.1930 dB    V 44.9964 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b11f85572be67abfb87b4c4f914f2b55,4af65c31fa927f6189df7499ffd81400,d6996a93e94fd477f4ff160b23f94c19]
    POC   10 TId: 0 ( I-SLICE, nQP 30 QP 30 )     158120 bits [Y 36.3880 dB    U 44.6886 dB    V 46.3408 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8e194332ce0f20dcf7ae882bb0274706,a09735981c06711ed51d6ad040caf8e2,b4326fc87dcf49a70e6b24585e6eaae4]
    POC   11 TId: 0 ( I-SLICE, nQP 30 QP 30 )     116288 bits [Y 36.5720 dB    U 43.9470 dB    V 40.8785 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4c73cf9dbc4741f62d6d9c40ad26db01,ba6a27f49c60b71f1067064d1cc13cb4,3f99b81c545806878d7fff123bef63c6]
    POC   12 TId: 0 ( I-SLICE, nQP 30 QP 30 )     295872 bits [Y 34.8145 dB    U 41.0880 dB    V 41.6288 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1fca1af2de892c20194500e407f84a19,f62b786ca3df8812515d367940ffef53,7a68a4dcad68988d05066913c07cf696]
    POC   13 TId: 0 ( I-SLICE, nQP 30 QP 30 )     374824 bits [Y 34.1166 dB    U 43.9649 dB    V 43.0775 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7855b9d7d6584cbce29f8449c7404781,5070282aaa1603d705e30d69ebc8cc7a,9f6bd743892aa8c394050c39030d63a4]
    POC   14 TId: 0 ( I-SLICE, nQP 30 QP 30 )     129096 bits [Y 37.1907 dB    U 44.3619 dB    V 41.8732 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1801020d9d1d6635aa59cb5edc1fd0ed,1d5ec1f1041a479336896a57e8582d50,d3765d10e045a0ea6f63fd9f0fbbefcb]
    POC   15 TId: 0 ( I-SLICE, nQP 30 QP 30 )     308944 bits [Y 35.5608 dB    U 40.6036 dB    V 42.1312 dB] [ET     3 ] [L0 ] [L1 ] [MD5:5b043f3533c58742eccd4533b73b8ea4,32d984cbb8f64e6576011864cee7787b,a205b9be1de1732d84ebc9f3ded2d123]
    POC   16 TId: 0 ( I-SLICE, nQP 30 QP 30 )     311376 bits [Y 35.0469 dB    U 40.7200 dB    V 41.1765 dB] [ET     3 ] [L0 ] [L1 ] [MD5:02f570e511e84c74f11e57853b8a079c,fa669536c74c546442be6e1f17c93553,f38951d6506acf6f79b59a302a528357]
    POC   17 TId: 0 ( I-SLICE, nQP 30 QP 30 )     185072 bits [Y 35.7988 dB    U 43.6320 dB    V 44.2994 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1e4d205fafb5594744f4ba509618cf19,8cb9c8ac1a85266b93c37b3d89bcf045,49e42251a988811101de8462e18619f5]
    POC   18 TId: 0 ( I-SLICE, nQP 30 QP 30 )     209064 bits [Y 35.5894 dB    U 41.3660 dB    V 41.9131 dB] [ET     3 ] [L0 ] [L1 ] [MD5:891469b4750f2a74e754b384cb7a36af,b1495cb5e3b3fb0e589f7b6d8d0e032b,b217d07851dc8c9dbfe081f40107517f]
    POC   19 TId: 0 ( I-SLICE, nQP 30 QP 30 )      90264 bits [Y 39.4201 dB    U 43.6056 dB    V 43.3606 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d410e27ff9fff9c61a7a1295d19aa599,22e988d2f72465be9b2b18d27ef83843,dd16886f2dda03d42da60134d39e620f]
    POC   20 TId: 0 ( I-SLICE, nQP 30 QP 30 )     113304 bits [Y 38.1449 dB    U 44.5797 dB    V 45.7932 dB] [ET     2 ] [L0 ] [L1 ] [MD5:05338b1fd16c0b86562868654f189ff9,82fdb9561a6214c01a04d25d84c4d04a,24f9184527a46893698f216545b9ef7d]
    POC   21 TId: 0 ( I-SLICE, nQP 30 QP 30 )     244568 bits [Y 36.0888 dB    U 42.5575 dB    V 44.4146 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d3dd4fbbc241e1f065ceb93e8242731c,f8cdbf21b734d783358b5ca6a6945a09,85523b12c8ac22f73efe45351879016c]
    POC   22 TId: 0 ( I-SLICE, nQP 30 QP 30 )     124216 bits [Y 37.7944 dB    U 43.7805 dB    V 45.1966 dB] [ET     2 ] [L0 ] [L1 ] [MD5:13b87bf65538218591a31b0114c487da,aa923515325c25d8df24ee9e3d78f20b,677204f32ae9157d9b25283ad94ccd14]
    POC   23 TId: 0 ( I-SLICE, nQP 30 QP 30 )     369432 bits [Y 34.4720 dB    U 41.4571 dB    V 41.1839 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c9aa9772942534845dd7386e30c6f49a,90c8b2d3dc60a3dce1a706559c087e95,18404f4ea1c6da3bc03262926f472ced]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   12993.5200   36.3369   43.0966   43.1861   37.4053  

# 29
    POC    0 TId: 0 ( I-SLICE, nQP 29 QP 29 )     240216 bits [Y 36.4004 dB    U 43.8125 dB    V 41.9663 dB] [ET     3 ] [L0 ] [L1 ] [MD5:47c05b1b13b792c4fc1f14ac8eced4cb,45bfa940ea45c5fddbe9dd490723e01d,19e8d2e765b7fa4bb0ff97043e7bf12c]
    POC    1 TId: 0 ( I-SLICE, nQP 29 QP 29 )     410928 bits [Y 35.5795 dB    U 40.5249 dB    V 40.9561 dB] [ET     4 ] [L0 ] [L1 ] [MD5:31ec7f43d341c7643d1b68777b24b334,815b2be776292b6ffd90b3fc1bf9cd10,297c928bd0302dd269442d3b84524ef9]
    POC    2 TId: 0 ( I-SLICE, nQP 29 QP 29 )     155616 bits [Y 37.1808 dB    U 45.5440 dB    V 40.8937 dB] [ET     3 ] [L0 ] [L1 ] [MD5:939bbdae36fb2d866759ecb7843eb6a3,771423ccd34f7a4b68f29ea5bbf3afb4,cd5f12acf891ee267b612d9b87399272]
    POC    3 TId: 0 ( I-SLICE, nQP 29 QP 29 )     134504 bits [Y 38.2682 dB    U 43.6307 dB    V 44.5207 dB] [ET     2 ] [L0 ] [L1 ] [MD5:c3939b4f655af1d0db34ce20a2951ab7,4e6039e88ab32d238db8b447ad5e14be,39cb9b433afdcf119f5ee04f408ed69e]
    POC    4 TId: 0 ( I-SLICE, nQP 29 QP 29 )     292712 bits [Y 36.1143 dB    U 43.3589 dB    V 43.7355 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a2081a6152b5637ad6665e999445e9a3,44172952640554a9178f155d00662756,a0350fc9e95e78219bb54410f3bbcea7]
    POC    5 TId: 0 ( I-SLICE, nQP 29 QP 29 )     117832 bits [Y 37.9807 dB    U 45.6811 dB    V 44.9298 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d77bb91d440dd88b133711a1f9f862bf,081ffc270f4a85c2c334c2940ec29fd7,c79089fe5c67d1376c23670089913641]
    POC    6 TId: 0 ( I-SLICE, nQP 29 QP 29 )     613592 bits [Y 34.3147 dB    U 40.3881 dB    V 42.6898 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5f1a6e46845cdbfaf48aec9bf0a509f8,8f3bd5bc0b4772eb411d1e1d4467d97a,c1aa6b81f873069ed84f7462af283e93]
    POC    7 TId: 0 ( I-SLICE, nQP 29 QP 29 )     156824 bits [Y 39.0992 dB    U 42.5384 dB    V 43.3291 dB] [ET     3 ] [L0 ] [L1 ] [MD5:923487da347aca2448560583b7306d50,692896f9b0a70aebbbac1a35ca54ad08,b901fc624751c90714c3049688031e51]
    POC    8 TId: 0 ( I-SLICE, nQP 29 QP 29 )     113928 bits [Y 39.2151 dB    U 44.1643 dB    V 44.8857 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d74f274fd63b8701efa6d1d0d255c9c7,560ac7a28a88947e7f75ccdf6099f77e,b2b7eadb8fcd2f29190a3480df2659e9]
    POC    9 TId: 0 ( I-SLICE, nQP 29 QP 29 )     167048 bits [Y 37.6869 dB    U 44.0812 dB    V 44.8561 dB] [ET     3 ] [L0 ] [L1 ] [MD5:06bf82721a65333add0b955d7169074e,167bb586d183528cf7589b7eba41b810,99669171ea413fa688810316fd4a0dcf]
    POC   10 TId: 0 ( I-SLICE, nQP 29 QP 29 )     180752 bits [Y 37.0844 dB    U 44.6006 dB    V 46.1826 dB] [ET     3 ] [L0 ] [L1 ] [MD5:29fafb576a99cd3d77bd92bb012e07d1,a1707757f4f8a5de7be5598aa725ea8c,cec12a34a3aeda343e64af43a12f2aa7]
    POC   11 TId: 0 ( I-SLICE, nQP 29 QP 29 )     134672 bits [Y 37.1552 dB    U 43.8261 dB    V 40.8351 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4f5b05cf074391afd01d7dc0effe159f,024d20b1c99367e698d5b8023232ad58,405a332251d0f93a11d2feaf9b198ad3]
    POC   12 TId: 0 ( I-SLICE, nQP 29 QP 29 )     331496 bits [Y 35.5524 dB    U 41.0736 dB    V 41.5219 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4c8de88aff9c6ac0da648a043ea16051,d6aab67ce5b3bf8614fdb1d9b35f23dc,6199a0d676feeea63f5110224d7a6532]
    POC   13 TId: 0 ( I-SLICE, nQP 29 QP 29 )     420256 bits [Y 34.9483 dB    U 43.9226 dB    V 43.0696 dB] [ET     3 ] [L0 ] [L1 ] [MD5:569a12e653f994a2513e654b8c2123fc,98cf2601313508f6f5c33e20ac30efdf,7545990b9ac06593bd22824b1355014d]
    POC   14 TId: 0 ( I-SLICE, nQP 29 QP 29 )     146072 bits [Y 37.8330 dB    U 44.2389 dB    V 41.9146 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d9a00bef11e304ca7698914901e84829,ee7929a58ac942e74d27fa3a9ee65c9d,597ca1bff4fac3be0b50711f32112c61]
    POC   15 TId: 0 ( I-SLICE, nQP 29 QP 29 )     337640 bits [Y 36.2974 dB    U 40.5637 dB    V 42.0761 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ba5d89516d7e1b4b990ebf1d80632881,b25258eee24e6037dcc0e279649dbd07,02c86d97d37a581dba74dab421abf53a]
    POC   16 TId: 0 ( I-SLICE, nQP 29 QP 29 )     342920 bits [Y 35.7631 dB    U 40.7255 dB    V 41.1615 dB] [ET     3 ] [L0 ] [L1 ] [MD5:89b3e4cf44290f31b6a679abfda8ce13,7d5f19cc83b8041f2f3afb3c7862ea4d,e546e8635556b833532a73dcea342ecd]
    POC   17 TId: 0 ( I-SLICE, nQP 29 QP 29 )     210920 bits [Y 36.5145 dB    U 43.5715 dB    V 44.1440 dB] [ET     3 ] [L0 ] [L1 ] [MD5:db6ab4359e30450715c920fa17c5ad00,49a27c0c3fb0249479bab33ff45363b2,cc732144e39d4063f9e8b267f7920f87]
    POC   18 TId: 0 ( I-SLICE, nQP 29 QP 29 )     237144 bits [Y 36.2937 dB    U 41.3808 dB    V 41.8538 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e0e7c46585fc041e5f5a9461fc273787,9a440e68b1e215d05e9e96c50655db30,f0c09a142606b041f57592fb8f763743]
    POC   19 TId: 0 ( I-SLICE, nQP 29 QP 29 )      98344 bits [Y 39.9432 dB    U 43.6600 dB    V 43.3263 dB] [ET     2 ] [L0 ] [L1 ] [MD5:1c1547ecdef789cb8442fb159d7ec6ee,b9f68f6056d31a2c0baa71310054d749,a74e61d6b3b549acaf91694255fefb95]
    POC   20 TId: 0 ( I-SLICE, nQP 29 QP 29 )     125528 bits [Y 38.6756 dB    U 44.5824 dB    V 45.8150 dB] [ET     3 ] [L0 ] [L1 ] [MD5:00fd172c412eced460fb3d723fb2db78,e3e9c19481a981862cb8f6b64d764f54,43a3f9280d7cb5186de1f6c394368860]
    POC   21 TId: 0 ( I-SLICE, nQP 29 QP 29 )     269904 bits [Y 36.8187 dB    U 42.5481 dB    V 44.4244 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d4d697ac02779bb432d3e6c161fd8f5c,8945ad3e5a8dbc9d959589b4514eb25b,d648fcaf2a4915b8fc7fc8338668ee86]
    POC   22 TId: 0 ( I-SLICE, nQP 29 QP 29 )     139872 bits [Y 38.5019 dB    U 43.7281 dB    V 44.9200 dB] [ET     2 ] [L0 ] [L1 ] [MD5:397081971ae3ffc4bc70c84b0e1257c8,7a83ee33286f7c55ebd35b5c38b07b54,c5962eba5e22c21b56a825e67c53e02b]
    POC   23 TId: 0 ( I-SLICE, nQP 29 QP 29 )     406656 bits [Y 35.2296 dB    U 41.5058 dB    V 41.2211 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7c288f5313ab3248e8fab91d47d8e730,13f1b663ec72a4b2585ead6c9dbdb22a,754b3145df9e27a201684fd3797f3488]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   14463.4400   37.0188   43.0688   43.1345   38.0363  

# 28
    POC    0 TId: 0 ( I-SLICE, nQP 28 QP 28 )     272936 bits [Y 37.1726 dB    U 44.4874 dB    V 42.4927 dB] [ET     4 ] [L0 ] [L1 ] [MD5:31ac8b1771b38f114de18bd017f92558,e9c2bcf0f37eee331571405230cc52fe,66cca3d6298af7f63ca93fd5544dc38d]
    POC    1 TId: 0 ( I-SLICE, nQP 28 QP 28 )     457136 bits [Y 36.4625 dB    U 41.2642 dB    V 41.6258 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e0ea1bfa90922635b0fc6e19973e3175,cda05e826fef1d9aed909c43e735d322,a33f8890c96d5ecabbfe21c22989e7e9]
    POC    2 TId: 0 ( I-SLICE, nQP 28 QP 28 )     181952 bits [Y 37.8695 dB    U 46.1652 dB    V 41.6722 dB] [ET     3 ] [L0 ] [L1 ] [MD5:9c3f3dbb01bfbebcc369f17d34b0d122,7e684a85b2fb3c1ff4a1ba6baf4f9eb6,0211713453a172909b54efad4e927f1b]
    POC    3 TId: 0 ( I-SLICE, nQP 28 QP 28 )     152048 bits [Y 38.8850 dB    U 44.1756 dB    V 44.9741 dB] [ET     3 ] [L0 ] [L1 ] [MD5:341e8bbe284acbb31c779f2742d01872,8675e03599b4651b1c8d6f43d89f473b,1050a1b210b391a6b0b970516b15776a]
    POC    4 TId: 0 ( I-SLICE, nQP 28 QP 28 )     328432 bits [Y 36.9296 dB    U 43.7977 dB    V 44.2891 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d06084cc66ad97400be3ce95a3f0f713,55c7640fed2f91c25cfa8270f264183c,16b73d9f1fc051907b5a120d3389d930]
    POC    5 TId: 0 ( I-SLICE, nQP 28 QP 28 )     136528 bits [Y 38.5887 dB    U 46.4060 dB    V 45.4168 dB] [ET     3 ] [L0 ] [L1 ] [MD5:fdd36a1d6fc06536a9a3456b659d6c98,abb4135f9d7450d66dce2869254db923,b2f3aa7da6dd7019c7c3d25a9bd8cdf2]
    POC    6 TId: 0 ( I-SLICE, nQP 28 QP 28 )     678304 bits [Y 35.2553 dB    U 40.8881 dB    V 43.2449 dB] [ET     4 ] [L0 ] [L1 ] [MD5:8080b63944720e21f0fad9044162bb01,25eb765af42608d6c857c09053436d61,df3b8bce504509aac29ec6e735f13e25]
    POC    7 TId: 0 ( I-SLICE, nQP 28 QP 28 )     173344 bits [Y 39.7775 dB    U 43.1625 dB    V 43.8587 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2abbeab1790bcbb19a16a4a7a3c07e2a,72180d85b40782f94f89886f20b3f6c4,8b857841e5afed6202589c733888b512]
    POC    8 TId: 0 ( I-SLICE, nQP 28 QP 28 )     129576 bits [Y 39.8787 dB    U 44.9418 dB    V 45.5497 dB] [ET     2 ] [L0 ] [L1 ] [MD5:a70953de4f475452d0fcaa6895c7dea6,a527eb784d65c283741bd50d6bcf7aa6,cf1c9476f0da03225e4362648663c570]
    POC    9 TId: 0 ( I-SLICE, nQP 28 QP 28 )     187560 bits [Y 38.3599 dB    U 44.5424 dB    V 45.1902 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a28429e27fcf8761e652a74eaaea15f6,4fb0b93557daedc3236aafabdd1e2384,02bb4f4f11bcb5df184e89801f29c200]
    POC   10 TId: 0 ( I-SLICE, nQP 28 QP 28 )     207368 bits [Y 37.8177 dB    U 45.0981 dB    V 46.6624 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c322938923c3225ccc4360bd2e0ce8f8,31e3a32c6c8961213812688b1af163ef,d76129c31f9962aa1060d8fd7523b95c]
    POC   11 TId: 0 ( I-SLICE, nQP 28 QP 28 )     159712 bits [Y 37.8125 dB    U 44.3912 dB    V 41.4423 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4e12464114e8f0080fafd4ca37bbda75,8173918cf879d68eb6fba81b4525a744,9260d80b911b120d18864f26fd3b5286]
    POC   12 TId: 0 ( I-SLICE, nQP 28 QP 28 )     374256 bits [Y 36.3476 dB    U 41.8028 dB    V 42.1480 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0b5b4d9a8194e20a34340816cd3757e7,b065ce10da71c6e76a0aaa2190c85513,bb0de1fa93c4cef7003a696ceaca44fc]
    POC   13 TId: 0 ( I-SLICE, nQP 28 QP 28 )     469232 bits [Y 35.8297 dB    U 44.5017 dB    V 43.5917 dB] [ET     4 ] [L0 ] [L1 ] [MD5:bb215d8fbfbb791bf23360eef0e6d093,3f176056802d9a84828ee52af6b88772,c031ed6b281c2bdb7c8880fa680a2ea5]
    POC   14 TId: 0 ( I-SLICE, nQP 28 QP 28 )     168208 bits [Y 38.5109 dB    U 44.9478 dB    V 42.4479 dB] [ET     3 ] [L0 ] [L1 ] [MD5:38ded5f642b5c89fc704850891256875,9a5294441225a7c2add343481e0c21c3,8d54b917c4ea3f74634ba25cacbf7214]
    POC   15 TId: 0 ( I-SLICE, nQP 28 QP 28 )     375248 bits [Y 37.1454 dB    U 41.2308 dB    V 42.7125 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4e41ea4b2763f37fd9a4b73ba277c96f,e462a51ad3a8f3aae2d5295c51ac5ad9,d1bd99bc16da2869dde8274ac553590e]
    POC   16 TId: 0 ( I-SLICE, nQP 28 QP 28 )     383616 bits [Y 36.5366 dB    U 41.2702 dB    V 41.7390 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0a4fc5b2a36abcb5a8a9c346b104c5fb,3b90bbeac29689c8231a440d8daa5cd9,ffab264742fae59055d4562e5caae3ba]
    POC   17 TId: 0 ( I-SLICE, nQP 28 QP 28 )     241384 bits [Y 37.2703 dB    U 44.0166 dB    V 44.5298 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e92b4b340cc9b3f587229202fc8117e5,6dcca1e8626078cf65f1664d61ee39fb,9e4c5fb6251974c564b4a6da345794d6]
    POC   18 TId: 0 ( I-SLICE, nQP 28 QP 28 )     271784 bits [Y 37.0630 dB    U 41.8271 dB    V 42.4220 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f1236e3a2ddad3135046863ecdce6b0a,2b602d8377ee70904a9c2ccb6ebdebd5,990439bf6b8854eeb596d5fd3ffd83e9]
    POC   19 TId: 0 ( I-SLICE, nQP 28 QP 28 )     111064 bits [Y 40.4902 dB    U 44.1954 dB    V 43.9205 dB] [ET     2 ] [L0 ] [L1 ] [MD5:d9987346a4b426d4ef8e36d20e8fb184,d61c54a9f39b48aed878951018c26b68,c1388402a1df33b68a5931c9174f3b8c]
    POC   20 TId: 0 ( I-SLICE, nQP 28 QP 28 )     139656 bits [Y 39.2750 dB    U 45.1631 dB    V 46.0554 dB] [ET     2 ] [L0 ] [L1 ] [MD5:95e898cbd1c79437d3433145f7f755c1,c6316af8b27de2391de02f635223ab67,e8e3c3f54c45f6822de956c1bae97e90]
    POC   21 TId: 0 ( I-SLICE, nQP 28 QP 28 )     300200 bits [Y 37.6110 dB    U 42.9665 dB    V 44.7876 dB] [ET     3 ] [L0 ] [L1 ] [MD5:32d72a869b70d31bf239b29b1495e9c3,80dcf85153e78cdd588ef4b0530a63d4,3814c19f2a42023fb4df88691fbedd6d]
    POC   22 TId: 0 ( I-SLICE, nQP 28 QP 28 )     159040 bits [Y 39.2984 dB    U 44.3150 dB    V 45.5687 dB] [ET     2 ] [L0 ] [L1 ] [MD5:2ea53cdb823d3b4999ae11c768d15952,e253b16b1799ec44fcd4446b15073e36,15423938fb01046413233175c464ebde]
    POC   23 TId: 0 ( I-SLICE, nQP 28 QP 28 )     450320 bits [Y 36.0204 dB    U 42.0935 dB    V 41.8439 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7e4850fec2277403ea38f93015a4d029,29509970f12fe61e6a338289ce4a4201,49f371887f842eb69200514a3a7aefbb]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   16272.2600   37.7587   43.6521   43.6744   38.7805  

# 27
    POC    0 TId: 0 ( I-SLICE, nQP 27 QP 27 )     307136 bits [Y 37.9021 dB    U 44.8274 dB    V 43.1017 dB] [ET     3 ] [L0 ] [L1 ] [MD5:be137c8d7c69d1466305e32350fa73dd,e62df9d306712f953ef9855a83f0baee,12c0ec1b3c70ea066478d929c70afc2a]
    POC    1 TId: 0 ( I-SLICE, nQP 27 QP 27 )     501752 bits [Y 37.2917 dB    U 41.6947 dB    V 42.1706 dB] [ET     4 ] [L0 ] [L1 ] [MD5:66e81f3b8d63b8294444ec30a2eca47d,2959820d04254fa156453f6e256c9209,d490fcffb4775c30caf71631e34f1684]
    POC    2 TId: 0 ( I-SLICE, nQP 27 QP 27 )     207216 bits [Y 38.4959 dB    U 46.7228 dB    V 42.2368 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d025c77969353c1336509fcbe16a7f8b,40512e0b201015739529589ccb1ff932,6304420a3ad94ce2d22f25391874636d]
    POC    3 TId: 0 ( I-SLICE, nQP 27 QP 27 )     168968 bits [Y 39.4392 dB    U 44.5989 dB    V 45.5066 dB] [ET     3 ] [L0 ] [L1 ] [MD5:555bf29f033270a4803f65e6ae941f88,9111b44163df4199b01dac6c19539e9d,5d7473b7c810c55fc047f78c1d192456]
    POC    4 TId: 0 ( I-SLICE, nQP 27 QP 27 )     364328 bits [Y 37.7014 dB    U 44.2024 dB    V 44.6679 dB] [ET     3 ] [L0 ] [L1 ] [MD5:45bb255e91ec28adefe0fa22c1f8a039,75114fa5018ce9408a079ecfd36bac1c,ce8de6ba2faa393481b3ce85efbefcb5]
    POC    5 TId: 0 ( I-SLICE, nQP 27 QP 27 )     156504 bits [Y 39.2230 dB    U 46.7747 dB    V 45.7668 dB] [ET     3 ] [L0 ] [L1 ] [MD5:aec87dfaaf85cff29e2bab0728cec30a,01203046690ca8a98180e2ce5794df21,793eb98c39a51e52b4bf2126bc0361eb]
    POC    6 TId: 0 ( I-SLICE, nQP 27 QP 27 )     746184 bits [Y 36.2387 dB    U 41.3636 dB    V 43.6642 dB] [ET     4 ] [L0 ] [L1 ] [MD5:628ac68b2e16134b3b84176d4b5f2161,66dd5cc7e00dfba41ab5b8ebaa0add9c,d9bc560cb71ba3f2cea3c3f7faa3d196]
    POC    7 TId: 0 ( I-SLICE, nQP 27 QP 27 )     191960 bits [Y 40.4698 dB    U 43.6705 dB    V 44.5009 dB] [ET     3 ] [L0 ] [L1 ] [MD5:5f1effb22cb05f8a78dea79d91eebbf9,1a0772b26c0b09b3f46555bb81318c1e,ef5f797ca02d7c96a91f9a53c2c104f3]
    POC    8 TId: 0 ( I-SLICE, nQP 27 QP 27 )     145976 bits [Y 40.5584 dB    U 45.4290 dB    V 46.2427 dB] [ET     2 ] [L0 ] [L1 ] [MD5:25911516ea0f7ea6a0eed1d5afd4ae69,eec9f204ac2afcdd15dc478e81e99014,364189d795b1be9fe84df9f20d011519]
    POC    9 TId: 0 ( I-SLICE, nQP 27 QP 27 )     212144 bits [Y 39.0360 dB    U 44.9694 dB    V 45.6374 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3f485f849191a290b3d432d7eb85e09f,ab5948926ddc0e0a1c9da41f44fce038,1b6fa704add8492e8904cfd76ee019e4]
    POC   10 TId: 0 ( I-SLICE, nQP 27 QP 27 )     235400 bits [Y 38.5566 dB    U 45.5616 dB    V 47.0771 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f840133924f72d77fce4ce443a885e50,b00bd9a00351115f03f9dc5ab5d367c8,7bdc2f3c36f6e125295d268731c36e18]
    POC   11 TId: 0 ( I-SLICE, nQP 27 QP 27 )     187064 bits [Y 38.4542 dB    U 44.7234 dB    V 41.9078 dB] [ET     3 ] [L0 ] [L1 ] [MD5:bd6618e126af413344ac287f7b50d1ac,d360f43aa50b3dd02938b25a692d5251,205c101e6780ba20d0fe69c0800e88a5]
    POC   12 TId: 0 ( I-SLICE, nQP 27 QP 27 )     420056 bits [Y 37.1724 dB    U 42.3221 dB    V 42.7536 dB] [ET     4 ] [L0 ] [L1 ] [MD5:633b8d489338c4d78fe0de2fb190f557,16252e18cd56dd285e81e7ec61dd8e88,93dab0d330a9ae5824a3a808255e8cb9]
    POC   13 TId: 0 ( I-SLICE, nQP 27 QP 27 )     523456 bits [Y 36.7598 dB    U 44.9437 dB    V 44.0334 dB] [ET     4 ] [L0 ] [L1 ] [MD5:474b2eded7903cad1af514f6fb19e34a,e7ddf2c9a6bda9c857570cb69269fdc8,7483d240c8413709d1b0b4866692d5c3]
    POC   14 TId: 0 ( I-SLICE, nQP 27 QP 27 )     191976 bits [Y 39.1468 dB    U 45.4758 dB    V 43.0663 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f2bf9b5dcb498951488189b03259d912,a02338ef135c044d354f211789aa890f,a8363b16c8cb1543c5cca22f03ca4723]
    POC   15 TId: 0 ( I-SLICE, nQP 27 QP 27 )     414408 bits [Y 37.9685 dB    U 41.8121 dB    V 43.1853 dB] [ET     3 ] [L0 ] [L1 ] [MD5:9e450567840837b1634b15fb91e6743d,15e3ba4066cdb571235a2dd7d886f052,15b3219680fa5f6abec84fc0a92f31e7]
    POC   16 TId: 0 ( I-SLICE, nQP 27 QP 27 )     424248 bits [Y 37.2740 dB    U 41.7103 dB    V 42.2078 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c55f3a6b592c2b417dd13ee687960909,e38a6bba511611c87321f2772f7331d4,9ae671f99ff9276c154fc06b33f17368]
    POC   17 TId: 0 ( I-SLICE, nQP 27 QP 27 )     273144 bits [Y 37.9905 dB    U 44.5420 dB    V 45.0812 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f25cd9d0dde9c3da768650041ceae29f,6d7f9e73bd014954ee45ae544512b59d,52987471cb8880f508700487a8a9b943]
    POC   18 TId: 0 ( I-SLICE, nQP 27 QP 27 )     308424 bits [Y 37.8097 dB    U 42.3756 dB    V 42.9100 dB] [ET     3 ] [L0 ] [L1 ] [MD5:84940e8e2c3828051b03f6c9bdb33164,5f18b25fbf59d692f90eead90b2d5f85,942971508e61a79a551597bab47f1234]
    POC   19 TId: 0 ( I-SLICE, nQP 27 QP 27 )     124000 bits [Y 40.9917 dB    U 44.7335 dB    V 44.4174 dB] [ET     2 ] [L0 ] [L1 ] [MD5:9197b70e61a400133e8759dff4158b38,50e88f4760f5f84fd4abd0e0a06fff27,05f53fbfef3826dd0d112c0dd3ab9c8c]
    POC   20 TId: 0 ( I-SLICE, nQP 27 QP 27 )     155544 bits [Y 39.8433 dB    U 45.5294 dB    V 46.4409 dB] [ET     3 ] [L0 ] [L1 ] [MD5:001bc635b1f08bc11916ae99e822876d,afb5f7a1ffb1f52e62ed29e712213fbf,aba11d5f68e7b4ee464066d0575e88e3]
    POC   21 TId: 0 ( I-SLICE, nQP 27 QP 27 )     332000 bits [Y 38.3690 dB    U 43.5050 dB    V 45.3413 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7950d334cecfc44bc40fc49a8821c33f,93e3ccbae805dbb225e74555ed3bdafc,db7a2a8f2a948a1a54154491405b9297]
    POC   22 TId: 0 ( I-SLICE, nQP 27 QP 27 )     181376 bits [Y 40.0583 dB    U 44.8564 dB    V 46.0845 dB] [ET     2 ] [L0 ] [L1 ] [MD5:3e1428d9db6647ad5e918c2778832358,84b3d02bccd630f38f3b0d412249ca1f,e279193765542205b7ac12f8327f9c4f]
    POC   23 TId: 0 ( I-SLICE, nQP 27 QP 27 )     496624 bits [Y 36.7993 dB    U 42.6064 dB    V 42.4057 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b398957b55e8a6a85758d5e56f26ed36,ba8ee584f98715ee3e527272cf0631bf,2d70ae78a641fc8edf00f0d7740b2ff9]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   18174.7200   38.4813   44.1229   44.1837   39.4997  

# 26
    POC    0 TId: 0 ( I-SLICE, nQP 26 QP 26 )     343672 bits [Y 38.6490 dB    U 45.4161 dB    V 43.7191 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d06c46aa117a6257e0b4ca881d6fbbfc,995e01f86792e125c3f31b4a207b3e12,b9ce5c9760a45a3948311ed47e070576]
    POC    1 TId: 0 ( I-SLICE, nQP 26 QP 26 )     549672 bits [Y 38.1293 dB    U 42.3107 dB    V 42.7742 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9cb4be3d25b8e0a2e101173d7e200529,1e4b729e911c47c0f6192e014665ab5d,28c5db851672998c0cac077078987dd7]
    POC    2 TId: 0 ( I-SLICE, nQP 26 QP 26 )     236448 bits [Y 39.1443 dB    U 47.0985 dB    V 42.8249 dB] [ET     4 ] [L0 ] [L1 ] [MD5:70e85db7bef000ad30ab61ede3d0f449,fa0507595c26fa20bd3c150829da32f7,0a8feb439b03418801a40740eff9a2d1]
    POC    3 TId: 0 ( I-SLICE, nQP 26 QP 26 )     188896 bits [Y 40.0391 dB    U 45.0889 dB    V 45.9041 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8f1f0a032b04f31343969deac70ba2ed,3becd2e20b5614b9a0bbe8848c3a0196,5e5a4dd82c9e847b89bf3615f0fc37d8]
    POC    4 TId: 0 ( I-SLICE, nQP 26 QP 26 )     404128 bits [Y 38.5018 dB    U 44.5674 dB    V 45.1719 dB] [ET     4 ] [L0 ] [L1 ] [MD5:0c5b5b3ba86110fb44032007f4ff1ff7,2739733766e052a7d68b77530148bfd3,d7ba69d64d3b02001ddab0b51370114b]
    POC    5 TId: 0 ( I-SLICE, nQP 26 QP 26 )     179840 bits [Y 39.8438 dB    U 47.2985 dB    V 46.3618 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ec72616ea8125a54ff4f4d592cb665e2,1a89ac59802acd57c4c9b56e17891d69,af9880ee36cc2daaf3153caf9791e7b9]
    POC    6 TId: 0 ( I-SLICE, nQP 26 QP 26 )     814712 bits [Y 37.2004 dB    U 41.8887 dB    V 44.1359 dB] [ET     4 ] [L0 ] [L1 ] [MD5:0864693dd05cc8af69bb14de8101857d,4d6c11acaf6d555885cac9083838c9be,a3511b9d65e6850a6f785ad8334d1b73]
    POC    7 TId: 0 ( I-SLICE, nQP 26 QP 26 )     211216 bits [Y 41.1099 dB    U 44.4747 dB    V 45.0000 dB] [ET     3 ] [L0 ] [L1 ] [MD5:43302f67231a40d1e8998e10bb4b106a,3085c2318b6877c25dcd794a8768f665,f6ad06618e0125953b9d707521b2cc16]
    POC    8 TId: 0 ( I-SLICE, nQP 26 QP 26 )     163344 bits [Y 41.2036 dB    U 46.0620 dB    V 46.6127 dB] [ET     3 ] [L0 ] [L1 ] [MD5:430fa2f6b48d41d67150fd0d6f3fe553,9ddb5fbf35672a88aa7c344c1867628b,061d63a3cafb1443264c3596c0d8b4aa]
    POC    9 TId: 0 ( I-SLICE, nQP 26 QP 26 )     236856 bits [Y 39.6747 dB    U 45.4874 dB    V 46.2126 dB] [ET     3 ] [L0 ] [L1 ] [MD5:973f32f6a0905f800fef07d388861e50,6b9f24a97d65fa7cd68a65735c43e2b1,cef06284d60afeb0f142d9c50600460e]
    POC   10 TId: 0 ( I-SLICE, nQP 26 QP 26 )     265416 bits [Y 39.2659 dB    U 46.0619 dB    V 47.5099 dB] [ET     3 ] [L0 ] [L1 ] [MD5:59a3fa6132144ff037f290a8ca670c8c,eb0baafceb8bc90a3aabc68393409e79,6966d64b80048a3388f2e16cfeee9bb4]
    POC   11 TId: 0 ( I-SLICE, nQP 26 QP 26 )     217056 bits [Y 39.1059 dB    U 45.2261 dB    V 42.4444 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b4a79a29f7f043c83c3ef6a02df9a7d7,e0a3bfb2403ec4af1bf65714fb07ccfc,3f4633609f4cfec0569a874b72cd213d]
    POC   12 TId: 0 ( I-SLICE, nQP 26 QP 26 )     467128 bits [Y 37.9569 dB    U 42.9495 dB    V 43.3221 dB] [ET     4 ] [L0 ] [L1 ] [MD5:6afdbbbcfdb8c79a8ec7b5a55fe7e6f4,3504a3290dde01b84fd35992ac2ea2af,f0f57e3b67ed356df68f14ef580b81b9]
    POC   13 TId: 0 ( I-SLICE, nQP 26 QP 26 )     576416 bits [Y 37.6447 dB    U 45.1765 dB    V 44.3649 dB] [ET     4 ] [L0 ] [L1 ] [MD5:832a42ee0584e8660bf366f7f12dfc65,057020953ba17a62509a9866e951d825,ddc3c99b5fff37235dbaad7be1569a3f]
    POC   14 TId: 0 ( I-SLICE, nQP 26 QP 26 )     217248 bits [Y 39.8002 dB    U 45.9669 dB    V 43.6032 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e7f0048ba0c78fdf4cf9c37761c642fd,b11a3f05729acf836a7a6782a7156637,e73bad2d94218f35488d54d920fc9b64]
    POC   15 TId: 0 ( I-SLICE, nQP 26 QP 26 )     454296 bits [Y 38.7651 dB    U 42.4271 dB    V 43.7703 dB] [ET     3 ] [L0 ] [L1 ] [MD5:758da63952928509e92da63d0dadf55f,0f27469869bbc10d96993cc379ce57df,af8fb3e97958e3aa42bc77b363731bb3]
    POC   16 TId: 0 ( I-SLICE, nQP 26 QP 26 )     467304 bits [Y 37.9920 dB    U 42.2364 dB    V 42.6635 dB] [ET     4 ] [L0 ] [L1 ] [MD5:344c955229b170f7353496ad7e9cb265,502d5162c99188f805f3f26edd7edf2a,2ca4c8e095155a183d3effbd39f82cda]
    POC   17 TId: 0 ( I-SLICE, nQP 26 QP 26 )     309480 bits [Y 38.7723 dB    U 44.9024 dB    V 45.4326 dB] [ET     3 ] [L0 ] [L1 ] [MD5:eb1e3a3e1a97077b0a1af75e018d1976,699f1797737254ab4379bb535c033a61,30afe6c1cfa0a9761d04935acd799391]
    POC   18 TId: 0 ( I-SLICE, nQP 26 QP 26 )     348912 bits [Y 38.5739 dB    U 42.8673 dB    V 43.3798 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7761f2ab6ba95f7ed56f9f2a9b0f4e8e,81a94eb704a28b703619e0e3d4c2d1cc,74840f23419eaf908e59f455b2c948b7]
    POC   19 TId: 0 ( I-SLICE, nQP 26 QP 26 )     138256 bits [Y 41.4861 dB    U 45.2978 dB    V 44.9883 dB] [ET     2 ] [L0 ] [L1 ] [MD5:5c0044c071ff7c92d5ef7afaa618349a,9523cc9e7b26ff79fbfaf4899cf7b5c9,bf4ee8710b104e5857373a8a1a69e672]
    POC   20 TId: 0 ( I-SLICE, nQP 26 QP 26 )     172592 bits [Y 40.3573 dB    U 46.0799 dB    V 47.0598 dB] [ET     2 ] [L0 ] [L1 ] [MD5:da3fd3228ad9093908ad2eb1c6ea92db,0e0c757387b5d8b4e3cceb8a608e0bde,e94f7212b1c5842cf8cdfff41fc03934]
    POC   21 TId: 0 ( I-SLICE, nQP 26 QP 26 )     365224 bits [Y 39.1233 dB    U 43.8417 dB    V 45.7459 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2ddcc3b75d0e45a209dbea514252d4ce,4a5f26bb11b1aaeb72dfc8a51b95b689,730b80bd97185a1f798291543ea1326c]
    POC   22 TId: 0 ( I-SLICE, nQP 26 QP 26 )     203344 bits [Y 40.8210 dB    U 45.3234 dB    V 46.6818 dB] [ET     2 ] [L0 ] [L1 ] [MD5:4afbf2b9909182e283258930e596c5aa,773b27e2b6a9677d54a31a6a09633537,86283fffa2b6a94a2e9cba6bd777f197]
    POC   23 TId: 0 ( I-SLICE, nQP 26 QP 26 )     545448 bits [Y 37.6045 dB    U 43.0182 dB    V 42.9650 dB] [ET     4 ] [L0 ] [L1 ] [MD5:969b76ad31e607c837e6942fc39f8ed3,53d30dc4ca9afc5d2c97bbc123f0fd00,91a690eb4f1760564eab9e723cca5e5e]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   20192.2600   39.1985   44.6278   44.6937   40.2134  

# 25
    POC    0 TId: 0 ( I-SLICE, nQP 25 QP 25 )     387024 bits [Y 39.4852 dB    U 45.9799 dB    V 44.3165 dB] [ET     4 ] [L0 ] [L1 ] [MD5:c06201e86a75d965a655fb9bfead5f6d,853b20153278098f3d761d5e84e524b5,bd0b96eb6e5572a1432d54841789ec40]
    POC    1 TId: 0 ( I-SLICE, nQP 25 QP 25 )     603368 bits [Y 39.0346 dB    U 42.8450 dB    V 43.3879 dB] [ET     4 ] [L0 ] [L1 ] [MD5:36f950afe5f5b8b4531463c895018949,56bc67293dcf7824123fb2e8ade586c6,998bf4e1671df664fc071f2b6f729670]
    POC    2 TId: 0 ( I-SLICE, nQP 25 QP 25 )     271232 bits [Y 39.8833 dB    U 47.4735 dB    V 43.3878 dB] [ET     3 ] [L0 ] [L1 ] [MD5:500d4db46d6e2b395bf0daca1598d2a7,fcabf25fb3f4f157e0926b100dfbb3fa,02e7c7a989654736cbbbafa2010e08c8]
    POC    3 TId: 0 ( I-SLICE, nQP 25 QP 25 )     211824 bits [Y 40.6146 dB    U 45.6153 dB    V 46.3453 dB] [ET     3 ] [L0 ] [L1 ] [MD5:640033e21720a47efb2f8190f7e604ed,afc4d183050c235580305675e40fc6b1,8ba225e660704bba31a48f1f9aaaa523]
    POC    4 TId: 0 ( I-SLICE, nQP 25 QP 25 )     448800 bits [Y 39.3561 dB    U 45.1032 dB    V 45.7048 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1462f87c64d20d031cee5838f5fae821,cb0ca2a70cc56ec25305c449c5ad510e,f44c39a1a822972542bcac88fde95ae6]
    POC    5 TId: 0 ( I-SLICE, nQP 25 QP 25 )     206768 bits [Y 40.5180 dB    U 47.4802 dB    V 46.8656 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e1f7160d20a7f7860bde5671e07d2c4e,233e749b2e1f093e3f6b0505da2da1c3,3e6dff074befaae40f4cf52d3913b3bc]
    POC    6 TId: 0 ( I-SLICE, nQP 25 QP 25 )     889600 bits [Y 38.2534 dB    U 42.3391 dB    V 44.5862 dB] [ET     4 ] [L0 ] [L1 ] [MD5:a8e8da341419cc22681794b02aa21782,0c341bf7a41fa7f3f9cbcf16d521816c,4a4da6029957360a55015ec19268d9d3]
    POC    7 TId: 0 ( I-SLICE, nQP 25 QP 25 )     233096 bits [Y 41.8296 dB    U 45.0357 dB    V 45.6111 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8eb5403896762f07be342d3299ec99bc,becf050771781cb65aaa26e55f05526f,895dc58511267b09a14df62120d736ba]
    POC    8 TId: 0 ( I-SLICE, nQP 25 QP 25 )     183960 bits [Y 41.8991 dB    U 46.5739 dB    V 47.2630 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ad6174d6b05d3c1d5da4c72173631c0c,d66b5bdd02d18f838fbad28b38c10b29,2c49c0b3ed2420205e084ffdedb15c4c]
    POC    9 TId: 0 ( I-SLICE, nQP 25 QP 25 )     265944 bits [Y 40.3427 dB    U 45.9776 dB    V 46.5913 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c5af44f89c740d5923fd40da38e42aa7,be1aa0cd5447a27e27201a3d82ce0e56,ae9ed0a170f4ff6367e7eaff49833e65]
    POC   10 TId: 0 ( I-SLICE, nQP 25 QP 25 )     299840 bits [Y 40.0640 dB    U 46.4813 dB    V 48.1106 dB] [ET     3 ] [L0 ] [L1 ] [MD5:57513ec407dc33fb85cfe497719bebf4,35b5ee54d2fb24a44535347985571004,15967f3f48aa757318a7ef763d2c673d]
    POC   11 TId: 0 ( I-SLICE, nQP 25 QP 25 )     253288 bits [Y 39.8255 dB    U 45.6498 dB    V 42.9990 dB] [ET     3 ] [L0 ] [L1 ] [MD5:743c8e25715320f5f59d856b5c8e9bce,149d2be9dda92942a0cde02f589c642c,08e8392606bf52b87928d6f269432157]
    POC   12 TId: 0 ( I-SLICE, nQP 25 QP 25 )     522584 bits [Y 38.8262 dB    U 43.7092 dB    V 44.0930 dB] [ET     4 ] [L0 ] [L1 ] [MD5:673c5856888a529c6c0388d2b66a6dcf,f2ad9ad4a714a2d803795d5fad2b7e2f,5688bf86e3c68a6f706640a74f9987e9]
    POC   13 TId: 0 ( I-SLICE, nQP 25 QP 25 )     637504 bits [Y 38.6221 dB    U 45.8059 dB    V 45.1290 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7fe61cc82e7e34a798af4566160b4eb3,9d604202b4f18dbae461970b067f4ced,18890e3711e81f568fa35c7664a1e097]
    POC   14 TId: 0 ( I-SLICE, nQP 25 QP 25 )     247344 bits [Y 40.5173 dB    U 46.3315 dB    V 44.3075 dB] [ET     3 ] [L0 ] [L1 ] [MD5:124f20cdb4e810354b076fc6034ef01f,83bda2d7f07b6fc5d3590b94ef197806,ddf7f4241704e96cf1ec3da66a4a3ba9]
    POC   15 TId: 0 ( I-SLICE, nQP 25 QP 25 )     502880 bits [Y 39.6753 dB    U 43.1295 dB    V 44.2962 dB] [ET     3 ] [L0 ] [L1 ] [MD5:550bfd399f835f428aaede607b224eb7,6e60737286da03bf549ba5d0e3674258,da90752ac46eee14c904d7b361847c32]
    POC   16 TId: 0 ( I-SLICE, nQP 25 QP 25 )     518768 bits [Y 38.7879 dB    U 42.8037 dB    V 43.1013 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0b6352180ede678590d2cfeb130bc362,e30c4a383a985af215b11774104891ba,704f553ff6e50d21c05b843c8f698dac]
    POC   17 TId: 0 ( I-SLICE, nQP 25 QP 25 )     347976 bits [Y 39.5511 dB    U 45.3342 dB    V 45.9556 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e14f27ec2adb74de88b130099b72f3a5,98641d4063fcc01b99513d573cb9c9b6,a7cf5b6d0c81be75c6b93bc83485b523]
    POC   18 TId: 0 ( I-SLICE, nQP 25 QP 25 )     392472 bits [Y 39.3728 dB    U 43.3664 dB    V 43.8332 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4288aef9a5ee0a74f1f7a35d6a69d330,69bcda118fc3ad769e552079abe5ba5b,444536ea02e71b4ab01a7e3fce59c0db]
    POC   19 TId: 0 ( I-SLICE, nQP 25 QP 25 )     155464 bits [Y 42.0074 dB    U 45.7419 dB    V 45.5366 dB] [ET     2 ] [L0 ] [L1 ] [MD5:ee0617ec2fe75a45b5ab4a86b6b02dab,b387fa86a09f6b0055e8b2ecb4ab0ee3,725c108de14912bc5450da82781aa637]
    POC   20 TId: 0 ( I-SLICE, nQP 25 QP 25 )     192016 bits [Y 40.9006 dB    U 46.5507 dB    V 47.4509 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ffc789a09f1a829ec594e4af09b2f164,21b3c1f687603cc0262bee5f1bea3efe,601bdc8e6d7bf95c751fac4b588193a3]
    POC   21 TId: 0 ( I-SLICE, nQP 25 QP 25 )     403304 bits [Y 39.9087 dB    U 44.4200 dB    V 46.1986 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e82552296c05daf474c38e8e45b23e24,92ab944db6131f9894b706d693f8d820,f517aeec873c735c550162f31b869a48]
    POC   22 TId: 0 ( I-SLICE, nQP 25 QP 25 )     227256 bits [Y 41.6216 dB    U 45.6854 dB    V 47.0607 dB] [ET     2 ] [L0 ] [L1 ] [MD5:0ed4059bed751244c90ae66a49103ecf,1c98f493c19c885e13f2b337b15e6f20,036347d7f7eea1c9175ba098833a327c]
    POC   23 TId: 0 ( I-SLICE, nQP 25 QP 25 )     601144 bits [Y 38.4423 dB    U 43.5753 dB    V 43.5644 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9b433f138ff43ae5b86852647457dbef,9a8f096a2b493555fdf8900e1db78cfe,bb1c09336df23f3bbc1562d48618b6e4]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   22508.6400   39.9725   45.1253   45.2373   40.9778  

# 24 
    POC    0 TId: 0 ( I-SLICE, nQP 24 QP 24 )     431552 bits [Y 40.3150 dB    U 46.2934 dB    V 44.7983 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f9e7f7450319b587aebf9faa85e73f37,dac9cbce5a8e876e0e20ef0f530c59fe,bdb16b985d8f070c1f125f3f7512d8cc]
    POC    1 TId: 0 ( I-SLICE, nQP 24 QP 24 )     657488 bits [Y 39.9277 dB    U 43.4208 dB    V 43.9916 dB] [ET     4 ] [L0 ] [L1 ] [MD5:084b6281bcaf129ede2b8d59735d2c9a,e4ca67bdcda2a4e880cb311e7ba30da7,d5b19b40e4d884da7522180a83c1f8fd]
    POC    2 TId: 0 ( I-SLICE, nQP 24 QP 24 )     309160 bits [Y 40.6268 dB    U 47.9120 dB    V 43.9901 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3e6dc03e43312c7cdbc8ffa843f01981,f1ca88d48cffd1dcf080f823baafc503,36ec9b025138c1fcf41f7f74fbfbc33b]
    POC    3 TId: 0 ( I-SLICE, nQP 24 QP 24 )     236376 bits [Y 41.2108 dB    U 46.2062 dB    V 46.9426 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1df51848b11155dfd2699642d2f822c9,6694de7d8127e483cb32d25287cf83bb,76d0dfab153e47dd89e2079a4ec20998]
    POC    4 TId: 0 ( I-SLICE, nQP 24 QP 24 )     496432 bits [Y 40.2232 dB    U 45.5784 dB    V 46.2209 dB] [ET     3 ] [L0 ] [L1 ] [MD5:05018fbda15e7551acac9d154e1d4c8a,cce0fed6b805f89f744817c054849e0e,a2684326290b3b418a857b43ea337072]
    POC    5 TId: 0 ( I-SLICE, nQP 24 QP 24 )     235944 bits [Y 41.2273 dB    U 47.9628 dB    V 47.3675 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7212066af4943d15c0a3381998561f44,16cbda2eda8b94cff1553458a145b461,b1d4ee0d37dbcb5e8d1fddc86d222222]
    POC    6 TId: 0 ( I-SLICE, nQP 24 QP 24 )     963072 bits [Y 39.2816 dB    U 42.8597 dB    V 44.9848 dB] [ET     5 ] [L0 ] [L1 ] [MD5:55251ea7218731dc17d2b6105c56d45d,a01d60752069c8997d637848b7aea97b,991b6469fd91c59583446994c9527ea7]
    POC    7 TId: 0 ( I-SLICE, nQP 24 QP 24 )     258104 bits [Y 42.5453 dB    U 45.5146 dB    V 46.1259 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6d2fb2c714257c3226c68214fd6eb282,fba0a4013239ba061556dde0cfa7a981,054c331bdbc1bdf9e878df3604e53e97]
    POC    8 TId: 0 ( I-SLICE, nQP 24 QP 24 )     205776 bits [Y 42.5763 dB    U 47.0685 dB    V 47.6915 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f3e0739f9058e6e748f70999e952cf9c,3cf6849e95fb2414876a5afd7f668df0,1bdd019cfcbf60ba2bbfa70fe92f35c5]
    POC    9 TId: 0 ( I-SLICE, nQP 24 QP 24 )     297160 bits [Y 41.0476 dB    U 46.4717 dB    V 47.0138 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f01ac7d0cc2044711e8287d7e4055668,4fd72eef4c5f562fe3de46a6dfb0fd55,13a13cf2d1c10e2efd2981773157a593]
    POC   10 TId: 0 ( I-SLICE, nQP 24 QP 24 )     333808 bits [Y 40.8333 dB    U 46.7353 dB    V 48.2240 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d0a0b9c7bfae6ae9fb0f9125c53fea3d,8378beaac853a25fae2a0d7556e2cf1b,8b714a61e7ec47c8b8580050f403047e]
    POC   11 TId: 0 ( I-SLICE, nQP 24 QP 24 )     291192 bits [Y 40.5776 dB    U 46.1091 dB    V 43.6739 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ac7f933b6c359a2fa8fd3417125f0299,ca03694be1b5fe1f1c7dd6f7b261a565,02211ac484171c340a6c937249142084]
    POC   12 TId: 0 ( I-SLICE, nQP 24 QP 24 )     578776 bits [Y 39.7085 dB    U 44.2871 dB    V 44.6274 dB] [ET     4 ] [L0 ] [L1 ] [MD5:180d8bed24654819c17f30ffbc6c9a3a,c9417c22dace4660458c24ee63538715,4e7929eaab4a9c8b53850260884ea7b1]
    POC   13 TId: 0 ( I-SLICE, nQP 24 QP 24 )     695208 bits [Y 39.5800 dB    U 46.0892 dB    V 45.5957 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9a1e62d8fc12bd8c30279e00f826f176,09731342f70487f282323f683ebbf616,0ff03d30c0f69e0b787325bcbcd3c89c]
    POC   14 TId: 0 ( I-SLICE, nQP 24 QP 24 )     280672 bits [Y 41.2705 dB    U 46.8869 dB    V 44.9533 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3bd76be8e5b7737c03b086dcae5236ad,e7d19894893fe1c155474e55d2b86b97,b5ead86625294af97f69881641886bdb]
    POC   15 TId: 0 ( I-SLICE, nQP 24 QP 24 )     548920 bits [Y 40.5410 dB    U 43.7280 dB    V 44.9983 dB] [ET     4 ] [L0 ] [L1 ] [MD5:3c9fb0e5ebda814a957023eba0c06022,4427a0543e3b9cd48706223829dace43,0d3a28e51c1340dd9a10d9b098cff46c]
    POC   16 TId: 0 ( I-SLICE, nQP 24 QP 24 )     569800 bits [Y 39.5496 dB    U 43.2888 dB    V 43.6698 dB] [ET     4 ] [L0 ] [L1 ] [MD5:6a58223ac309b0701f0f112b518af094,5a850266d4483b2e0557d06b40d5cd69,b6e4e88e90264dafdb52ce8963ba5893]
    POC   17 TId: 0 ( I-SLICE, nQP 24 QP 24 )     389128 bits [Y 40.3390 dB    U 45.7550 dB    V 46.3624 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ae70807b2edc09b5c1b3cf6825fa203b,c6c4dc6c3659c34d1e45a21a9183e80f,c9b6779751d7faf737b535256880b24f]
    POC   18 TId: 0 ( I-SLICE, nQP 24 QP 24 )     438560 bits [Y 40.1846 dB    U 43.9341 dB    V 44.4081 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2c219ed0b16695356e338123f8eb722e,172110f1c38e554b35df26aa70ab0268,90f5eba1492f81c80237cbea1cac2da4]
    POC   19 TId: 0 ( I-SLICE, nQP 24 QP 24 )     174024 bits [Y 42.5567 dB    U 46.3131 dB    V 46.0829 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0b47bc2eba9c528eae0cf5d8a92ce359,3c7682aeb249603973b093087e4d6bf0,e03d97c7007ceda626f2e96daf126b96]
    POC   20 TId: 0 ( I-SLICE, nQP 24 QP 24 )     212416 bits [Y 41.4153 dB    U 47.0135 dB    V 47.9474 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1698513b518f1f36d831a9c509bc6e56,c57bc83c12ff88d68e8e676af70c643d,24a5ede95558403f591c972625f364f2]
    POC   21 TId: 0 ( I-SLICE, nQP 24 QP 24 )     439816 bits [Y 40.6563 dB    U 44.8040 dB    V 46.7046 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8e86220f3c65849595a697adeaa78883,7ec4c1d12d397bcfd39bcd52d42bdef6,7b41ee911401e0af8ab883eb2e12c1a7]
    POC   22 TId: 0 ( I-SLICE, nQP 24 QP 24 )     254136 bits [Y 42.4263 dB    U 46.1713 dB    V 47.4980 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e331aeb07e757c8b77035240aabe55ba,a9345c201edb4cf4e80249c3bb377433,116edd11c6072c0a13136f337069509b]
    POC   23 TId: 0 ( I-SLICE, nQP 24 QP 24 )     658768 bits [Y 39.2906 dB    U 44.1593 dB    V 44.1615 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1414ec19bb7010f736d9bd4e685290f5,01e5493e5934ed84fa7e623049f78d7c,3d3269927b4b50f966b016af75d31cb2]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   24890.7200   40.7463   45.6068   45.7514   41.7333  

# 23
    POC    0 TId: 0 ( I-SLICE, nQP 23 QP 23 )     475912 bits [Y 41.1082 dB    U 46.8728 dB    V 45.2804 dB] [ET     4 ] [L0 ] [L1 ] [MD5:68b8bcc5fe2247af0d118ee4e3759007,2942e6f99054541f3c3afd0b40e88ce1,e36bc3df61c3b7038163403e17779a14]
    POC    1 TId: 0 ( I-SLICE, nQP 23 QP 23 )     713008 bits [Y 40.7777 dB    U 44.0470 dB    V 44.5293 dB] [ET     4 ] [L0 ] [L1 ] [MD5:06689e869c40f6da4bbdd25f56256c24,2ff4be974aed2e012cd233a7641ee8f5,48748adc21c0727a57edd5ffb0aea9cf]
    POC    2 TId: 0 ( I-SLICE, nQP 23 QP 23 )     347616 bits [Y 41.3120 dB    U 48.3106 dB    V 44.5919 dB] [ET     3 ] [L0 ] [L1 ] [MD5:5a04c9d79f61752ced53f1e6fcc428f8,b76597733109cf7d24cb37ae47e15c60,0f49969cdf1fac1504aea93e0f01e0cd]
    POC    3 TId: 0 ( I-SLICE, nQP 23 QP 23 )     260232 bits [Y 41.7124 dB    U 46.7487 dB    V 47.3853 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4876bfb1284053ca8ad39c2155ee37b0,c12835d70e782f41c65715eabb26b4a5,fbb792d58d0118d3075aa84310b3ef0d]
    POC    4 TId: 0 ( I-SLICE, nQP 23 QP 23 )     542640 bits [Y 41.0466 dB    U 46.0404 dB    V 46.6958 dB] [ET     4 ] [L0 ] [L1 ] [MD5:eb01e34368d0f9ddcd31dcf57ecb91b5,921decf176c8e683aec52cb15bba969a,3808eab8d5ec3f2c0bb88e747e06601d]
    POC    5 TId: 0 ( I-SLICE, nQP 23 QP 23 )     265760 bits [Y 41.8726 dB    U 48.3745 dB    V 47.9078 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7872b361599c8acaf722d53be72e396c,f38d6af8d5e86d09a28aca15801828ff,2098f864c735983617d829129a123bb9]
    POC    6 TId: 0 ( I-SLICE, nQP 23 QP 23 )    1033632 bits [Y 40.2526 dB    U 43.4073 dB    V 45.4909 dB] [ET     5 ] [L0 ] [L1 ] [MD5:bc3b5a410e08472c89ddb6c684ead654,6e815d71f9246b8fb07bf29b5d58881c,2d7c2acfa7aa047a6989976f36153548]
    POC    7 TId: 0 ( I-SLICE, nQP 23 QP 23 )     280560 bits [Y 43.1602 dB    U 46.1617 dB    V 46.6834 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e065298dcf3100b4a92cfab0f73a88cf,72a9707af08364cda21c7fde2eef9019,c3a45fb7349c66a271cbaaa51dfd7f66]
    POC    8 TId: 0 ( I-SLICE, nQP 23 QP 23 )     227544 bits [Y 43.2278 dB    U 47.6524 dB    V 48.3554 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e8471c80963a2738080d41b73998472e,b66d1a55bbfbf4164fab013290baf3bc,e1d703823a349c2345d02098a3a702b0]
    POC    9 TId: 0 ( I-SLICE, nQP 23 QP 23 )     328848 bits [Y 41.6665 dB    U 47.0012 dB    V 47.5248 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3828a316ae675cad9e8141352af5e982,c2ba269d8253a222d194dd1a2728f464,acee038934709ccf808b663db6047948]
    POC   10 TId: 0 ( I-SLICE, nQP 23 QP 23 )     372512 bits [Y 41.5959 dB    U 47.1918 dB    V 48.6506 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c6880f957294bdd8810bf2623e6e0bc4,c1914538bce8db6af2a225bc0fde1f44,6ad3a084867aa455f0ef03df6b066158]
    POC   11 TId: 0 ( I-SLICE, nQP 23 QP 23 )     332592 bits [Y 41.2973 dB    U 46.5985 dB    V 44.3229 dB] [ET     4 ] [L0 ] [L1 ] [MD5:c4ba84f876596e6db76eb5d830b562a1,0c192d358f39317335c5dd3520422d61,e5ab7b2a44e8e2a3355a4a956121677a]
    POC   12 TId: 0 ( I-SLICE, nQP 23 QP 23 )     636592 bits [Y 40.5608 dB    U 44.9089 dB    V 45.2214 dB] [ET     5 ] [L0 ] [L1 ] [MD5:117af6e36006084e8769ed6f7b03aedb,4093bd86a1a318c89c20295ba99b667e,5527c548a01bfff8c6de7bf04735fb0c]
    POC   13 TId: 0 ( I-SLICE, nQP 23 QP 23 )     753240 bits [Y 40.4825 dB    U 46.7625 dB    V 46.0349 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ff0d3ee87b18035bbca058892d839913,2b6736005b6b175006f668482fa0497a,ffb7878a45a73ea2c2cd2750a04a6fb0]
    POC   14 TId: 0 ( I-SLICE, nQP 23 QP 23 )     313912 bits [Y 41.9534 dB    U 47.2780 dB    V 45.6061 dB] [ET     3 ] [L0 ] [L1 ] [MD5:eca892ec08adea501a55bb372ecbc919,8365b3a96b246e66608cff6e445c9b36,44a51b9ff93f7eaa586c8f1376fa8f56]
    POC   15 TId: 0 ( I-SLICE, nQP 23 QP 23 )     593224 bits [Y 41.3237 dB    U 44.4160 dB    V 45.5654 dB] [ET     4 ] [L0 ] [L1 ] [MD5:28356c57c29856475cf72b1f1e2d34c2,ed509a3d65f70334cbeb6c9e4ce98e2b,e507c4e300504d49c776538114e4d759]
    POC   16 TId: 0 ( I-SLICE, nQP 23 QP 23 )     623176 bits [Y 40.2626 dB    U 43.9048 dB    V 44.2177 dB] [ET     4 ] [L0 ] [L1 ] [MD5:df272874f482dbcd7dc45a9f6e51f93e,666c158a7b60891b5c8473c5bc3b8f70,c9bf5da6f5a667f5612521253f42d35e]
    POC   17 TId: 0 ( I-SLICE, nQP 23 QP 23 )     428440 bits [Y 41.0531 dB    U 46.1372 dB    V 46.7246 dB] [ET     3 ] [L0 ] [L1 ] [MD5:9c1c2a07188e5af6ef1e06eda18d6da1,9c2b7d18846bdc8e3b281d72e2c42fad,710d949c4b130537d3b970575e371b98]
    POC   18 TId: 0 ( I-SLICE, nQP 23 QP 23 )     483728 bits [Y 40.9068 dB    U 44.4266 dB    V 44.8912 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1a4614dfcebe7a9f24067313daf079eb,0c7922c4a7204883eb3173a2fac9b33f,7c8c024d1f2e5665eecd57c3bafae333]
    POC   19 TId: 0 ( I-SLICE, nQP 23 QP 23 )     194744 bits [Y 43.0453 dB    U 46.8278 dB    V 46.6426 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1e07f117b476440e5d83ee0b2452948a,e8154a24b4f545e0e76c13baf09bb5a6,b683013efdf5e85461a82ad4f93e6abd]
    POC   20 TId: 0 ( I-SLICE, nQP 23 QP 23 )     232928 bits [Y 41.8448 dB    U 47.4418 dB    V 48.3824 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a5d52513301765e2c97be6b169399aee,08ddd48d4769843e0991366a30c9137b,891dbe89945dacb0d2178e5e805883b7]
    POC   21 TId: 0 ( I-SLICE, nQP 23 QP 23 )     477688 bits [Y 41.3389 dB    U 45.2859 dB    V 47.1816 dB] [ET     3 ] [L0 ] [L1 ] [MD5:58c94a509f9c6d29cb914a479e684a7b,2ad4694cff6e580cd6e2c1ebc3c72553,d4a0dac2a8812a16b9b88530d62cf838]
    POC   22 TId: 0 ( I-SLICE, nQP 23 QP 23 )     278784 bits [Y 43.1502 dB    U 46.5432 dB    V 48.0699 dB] [ET     3 ] [L0 ] [L1 ] [MD5:b92bda0d8b9158d92a3f61cad40bb067,61023ba93caf5a1acefe8e7e5bdc8b68,60cef0b48801be69407e28bf226c86f1]
    POC   23 TId: 0 ( I-SLICE, nQP 23 QP 23 )     718448 bits [Y 40.1122 dB    U 44.6342 dB    V 44.7584 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d08ccbfd420dd7c77b4b1ac886189a1b,c45e6018545d34478d01739de3f45f1c,1aa4d034fcb9b43cb91c454642cf1861]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   27289.4000   41.4610   46.1239   46.2798   42.4378  

# 22
    POC    0 TId: 0 ( I-SLICE, nQP 22 QP 22 )     526704 bits [Y 41.9294 dB    U 47.3660 dB    V 45.9288 dB] [ET     4 ] [L0 ] [L1 ] [MD5:3b614459d4ac64bacba3aea39d8d8249,9bf6ca44069da6c29315ae833a4f666f,cfb7a1ad5346223b9ddc45417ce4ebc5]
    POC    1 TId: 0 ( I-SLICE, nQP 22 QP 22 )     772824 bits [Y 41.6707 dB    U 44.6418 dB    V 45.0859 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9700e8501ef5893d00efb4578515246e,32791397d75ed6b2086ac87896427ec8,8044c5de0c1be9318286462c38954aab]
    POC    2 TId: 0 ( I-SLICE, nQP 22 QP 22 )     391512 bits [Y 42.0671 dB    U 48.5698 dB    V 45.1878 dB] [ET     4 ] [L0 ] [L1 ] [MD5:2d7dba864ce1b2476bee50334406fabb,123f5b74534c60c356ed0bd938ec795d,d237446366aa571a1e9b7d01419361a0]
    POC    3 TId: 0 ( I-SLICE, nQP 22 QP 22 )     288720 bits [Y 42.2253 dB    U 47.2749 dB    V 47.8206 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7807fe00eae3e27e70b0534ea3b343a2,44636412db6133a861c854f7f0fc180d,178f731372cfa384034c0a81565c3d3a]
    POC    4 TId: 0 ( I-SLICE, nQP 22 QP 22 )     594128 bits [Y 41.9084 dB    U 46.4568 dB    V 47.1707 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1b6674ae106e0807978fc4bd66e522d8,d7b914b03179c4075cfe78f804c0b820,057f80a8d856dc6c1f649186b5f7a7e9]
    POC    5 TId: 0 ( I-SLICE, nQP 22 QP 22 )     300232 bits [Y 42.5742 dB    U 48.6462 dB    V 48.3118 dB] [ET     3 ] [L0 ] [L1 ] [MD5:fa4512e7b64bd129b187165bed78f188,8f46ac36ded0f56f0d675e7fa1131b04,193b72bd616e5caecb3e5d6afc75c5cc]
    POC    6 TId: 0 ( I-SLICE, nQP 22 QP 22 )    1109448 bits [Y 41.3021 dB    U 43.9167 dB    V 45.9229 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8e6adb78deb67d107c55ed77e8f2e4ce,df3ce6ffd536113aa3efec0d329a14b0,4470b3f9b4d15c197a049e16ba48a52d]
    POC    7 TId: 0 ( I-SLICE, nQP 22 QP 22 )     306904 bits [Y 43.7978 dB    U 46.6263 dB    V 47.2784 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e42258e41d4b5451b39e5e596f185dfd,4ed23192f599acca6c3532c6c28ea391,5adbe69f96b4c5bd35f50bc80c3173aa]
    POC    8 TId: 0 ( I-SLICE, nQP 22 QP 22 )     252712 bits [Y 43.8762 dB    U 48.1071 dB    V 48.7834 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ea53258759e6069ec80ea2f55c89b742,9406352eecacdaaf3ace1da894dda8e8,d1ff978fcd4195753d3fbcdcda28e7d5]
    POC    9 TId: 0 ( I-SLICE, nQP 22 QP 22 )     365752 bits [Y 42.3401 dB    U 47.4637 dB    V 48.0303 dB] [ET     3 ] [L0 ] [L1 ] [MD5:22870d0796d41a462764e8b1ddea859d,94445a3b8db6911eea929a6bf73d19ea,2def6ff9c07c942a52d3fa4b74de61a4]
    POC   10 TId: 0 ( I-SLICE, nQP 22 QP 22 )     413168 bits [Y 42.3865 dB    U 47.6134 dB    V 48.8832 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2bef1f1b99fe9f7aaf1451fc26f91349,df9942ca5344fb2f86836037138dfb3e,a777479447701d5327d25864a3de1c7b]
    POC   11 TId: 0 ( I-SLICE, nQP 22 QP 22 )     378800 bits [Y 42.0656 dB    U 47.0781 dB    V 44.9257 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5228388e42f4de0557daf955879bf341,6a5934ad996583018097a2e84cf5af64,180163b38ba9d7e37c6e3c17c952e2ce]
    POC   12 TId: 0 ( I-SLICE, nQP 22 QP 22 )     697112 bits [Y 41.4329 dB    U 45.5555 dB    V 45.9470 dB] [ET     4 ] [L0 ] [L1 ] [MD5:c3edffc5942282daac7a315f724bfbb6,48c6263db3407b019896861bf69313f3,b24be51470702089e9851670a6b7d5a0]
    POC   13 TId: 0 ( I-SLICE, nQP 22 QP 22 )     816616 bits [Y 41.4512 dB    U 47.1764 dB    V 46.5663 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d6a3732b03d3f5e2be40d1327b9ca65a,f2d2dca434eccc110d5f79e9fd98d730,6e8497eea73a6d5735ecd2486a345894]
    POC   14 TId: 0 ( I-SLICE, nQP 22 QP 22 )     353040 bits [Y 42.6918 dB    U 47.7525 dB    V 46.1654 dB] [ET     3 ] [L0 ] [L1 ] [MD5:790bf7555c0862bb788e389678bc5b0f,03fcc09688e53bdb72a478cf8b6ae9b9,7a21cbcff89f642c30b71d816c0befa1]
    POC   15 TId: 0 ( I-SLICE, nQP 22 QP 22 )     645656 bits [Y 42.1933 dB    U 45.0375 dB    V 46.1537 dB] [ET     4 ] [L0 ] [L1 ] [MD5:cf6cfb890effbf49556c8b9ebe1a3403,f45e4950d34c45d4b678c530e9a10630,f447d22c62d43f661ed794ad49acb91a]
    POC   16 TId: 0 ( I-SLICE, nQP 22 QP 22 )     684416 bits [Y 41.0379 dB    U 44.4881 dB    V 44.8425 dB] [ET     4 ] [L0 ] [L1 ] [MD5:51c32a71172922f7861fd67252d00248,35735d0863eb20cbe88504ac163647a5,42fe020c953e22e9e4c44f9d6c191ade]
    POC   17 TId: 0 ( I-SLICE, nQP 22 QP 22 )     474216 bits [Y 41.8181 dB    U 46.5194 dB    V 47.1768 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c4f0241e817ef462cfe81d8cfef4cbf0,7a676aa9c172a89e7bebcce418a5406d,5cda4a67120837f1c6ae259cfa3f78c8]
    POC   18 TId: 0 ( I-SLICE, nQP 22 QP 22 )     536064 bits [Y 41.7107 dB    U 44.9586 dB    V 45.4292 dB] [ET     3 ] [L0 ] [L1 ] [MD5:550a40b85f013443c32ddd22723fff1c,46ee6a1e5a354e4c0142d2a7e6bce7b9,29186031bf05a285ffae1c4363d1e75a]
    POC   19 TId: 0 ( I-SLICE, nQP 22 QP 22 )     216136 bits [Y 43.5002 dB    U 47.2517 dB    V 47.2624 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f3d040fc60d4a58017b089a3d55631b7,79f3db6ee2ff5b35666155a5d5173846,d32b0c0f21652f9345057417e5d88f39]
    POC   20 TId: 0 ( I-SLICE, nQP 22 QP 22 )     257152 bits [Y 42.3005 dB    U 47.9507 dB    V 48.7138 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7c772acc81b8a70ed9a54df0496a22a2,c0187e26aced1ec90e15d79b0d0c9695,11e61871b8b6f3e4dcf1f44b52378d7f]
    POC   21 TId: 0 ( I-SLICE, nQP 22 QP 22 )     519552 bits [Y 42.0397 dB    U 45.8211 dB    V 47.5764 dB] [ET     3 ] [L0 ] [L1 ] [MD5:be1fb6bc7f8c15114c5f87c5d5942448,dfa725df6986ad042baa4742889adb8f,b7d8019401508a23913e05401ec1ad5d]
    POC   22 TId: 0 ( I-SLICE, nQP 22 QP 22 )     308232 bits [Y 43.9091 dB    U 47.0873 dB    V 48.5743 dB] [ET     3 ] [L0 ] [L1 ] [MD5:eb6e1fc27fff930a86f9bb848c2b399b,5b20ee385c0b094f778141d29330867d,95ab0f88aed5da4040ef760bd511fbbb]
    POC   23 TId: 0 ( I-SLICE, nQP 22 QP 22 )     784152 bits [Y 40.9948 dB    U 45.1990 dB    V 45.2832 dB] [ET     4 ] [L0 ] [L1 ] [MD5:2d1e6070dbbc9759e35c1822d776e56b,efa7d402c14f69a27f92a2f7848599c7,d5b38562c8fa54355228ebde833056ea]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   29983.1200   42.2176   46.6066   46.7925   43.1735  

# 21
    POC    0 TId: 0 ( I-SLICE, nQP 21 QP 21 )     578616 bits [Y 42.7555 dB    U 47.8700 dB    V 46.4409 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e767d80d4ab44163af4e99020f7ca8be,1ce2d6d80af724429206252a4af83e9a,f6a446ad0b72fc8006542c91184306ba]
    POC    1 TId: 0 ( I-SLICE, nQP 21 QP 21 )     834216 bits [Y 42.5543 dB    U 45.2046 dB    V 45.6696 dB] [ET     5 ] [L0 ] [L1 ] [MD5:819721b1fe4a49d65daf28f83e152153,bfda119b1768d8afa422d5b2109565fc,3a4c72918903766bdab94a60c5964dd6]
    POC    2 TId: 0 ( I-SLICE, nQP 21 QP 21 )     437648 bits [Y 42.7942 dB    U 48.9177 dB    V 45.8510 dB] [ET     4 ] [L0 ] [L1 ] [MD5:0c38d24cd4bde68b744f08c0ddf78c29,fe2f5e38f2abd5a87420dd280473f7a1,1b6ae0672c8fae51cb776521c147740b]
    POC    3 TId: 0 ( I-SLICE, nQP 21 QP 21 )     322976 bits [Y 42.7615 dB    U 47.7487 dB    V 48.3023 dB] [ET     3 ] [L0 ] [L1 ] [MD5:bb43923570bf2163a4bbabfe513f62d7,ba9ea3b7f782735723535cedae8b7b8a,32239fea074a393be39136f2b4c65802]
    POC    4 TId: 0 ( I-SLICE, nQP 21 QP 21 )     645880 bits [Y 42.7495 dB    U 46.9753 dB    V 47.6716 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d603efb37924286610e25dc8413a24c6,14cb8a787faff12a3c89f774c31bb67b,d1a4745cbfe60db22af4cc0cb1b28429]
    POC    5 TId: 0 ( I-SLICE, nQP 21 QP 21 )     337336 bits [Y 43.2510 dB    U 49.1162 dB    V 48.7733 dB] [ET     3 ] [L0 ] [L1 ] [MD5:03cc23886b66b9f12622c0b42aeb1bc8,955e847467ea7f4ccb50b11a5f6ce0f5,c6cf173077fcd4d7ed80541bd8112a18]
    POC    6 TId: 0 ( I-SLICE, nQP 21 QP 21 )    1185112 bits [Y 42.3329 dB    U 44.4409 dB    V 46.3320 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e1a1031e8829758587c032311af48c7b,6ece59c1d560cf706859c422951bb0fe,92483d396aaecfdcd9baee5bae46b322]
    POC    7 TId: 0 ( I-SLICE, nQP 21 QP 21 )     335664 bits [Y 44.4204 dB    U 47.2401 dB    V 47.9192 dB] [ET     3 ] [L0 ] [L1 ] [MD5:757c6410307eeb7a7f21cdda224af349,33a45ce34d689783ad5e1a5727d09f35,fb4789e888ec34fffb2cb678dbbfa121]
    POC    8 TId: 0 ( I-SLICE, nQP 21 QP 21 )     279600 bits [Y 44.5263 dB    U 48.6540 dB    V 49.2216 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a4e7d2c37fc383f1f724c966b9fd727c,adb247c918f495111e4e877a3a9f2fc3,12ed20272e5b15d091516e50e6f1f432]
    POC    9 TId: 0 ( I-SLICE, nQP 21 QP 21 )     405064 bits [Y 43.0012 dB    U 47.9284 dB    V 48.5345 dB] [ET     4 ] [L0 ] [L1 ] [MD5:c6c84f0dba476bf4e016f1c98823b714,3de8f601b74cfea3fc39a5c1a0e01b17,dfdda602c3ca2877f6c905fa6fab7dea]
    POC   10 TId: 0 ( I-SLICE, nQP 21 QP 21 )     457272 bits [Y 43.1925 dB    U 47.9914 dB    V 49.2203 dB] [ET     4 ] [L0 ] [L1 ] [MD5:de15a35a5e271e599bc433b05c244be8,a1c3d7e3cad29ddf9a3c61f66a1319aa,eda6d58722c5d526e5a8a8b99b35f0d1]
    POC   11 TId: 0 ( I-SLICE, nQP 21 QP 21 )     428048 bits [Y 42.8102 dB    U 47.5638 dB    V 45.5103 dB] [ET     3 ] [L0 ] [L1 ] [MD5:94b4101766da00c378a5a27a3d9445c1,8b7315145d7123328d57ebb450781d2c,82cf91e82bacf8486d5d3baf494e7740]
    POC   12 TId: 0 ( I-SLICE, nQP 21 QP 21 )     762792 bits [Y 42.3152 dB    U 46.1998 dB    V 46.5364 dB] [ET     4 ] [L0 ] [L1 ] [MD5:54176b51627f906bc1970a79a8fd6ba5,dd8514dd1097b97e272c45216aef6f1d,d2ff84deb476eecf82d8986f691b793c]
    POC   13 TId: 0 ( I-SLICE, nQP 21 QP 21 )     879552 bits [Y 42.4011 dB    U 47.6601 dB    V 47.0245 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b5e68d0ef3844b3bb98ff1f5371dfcac,af7c1344d3a4553e547d0aac3735079d,54e2bf311a02c691c3b9f253749dab1f]
    POC   14 TId: 0 ( I-SLICE, nQP 21 QP 21 )     392920 bits [Y 43.4267 dB    U 48.1773 dB    V 46.7832 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e0a1dd6db6027c2c52fb2bcdf0bc2359,81bd04f9724eaae9ed188b702e01f116,b783c369f7bfe3293e2263b176f669ca]
    POC   15 TId: 0 ( I-SLICE, nQP 21 QP 21 )     698672 bits [Y 43.0212 dB    U 45.6922 dB    V 46.6893 dB] [ET     4 ] [L0 ] [L1 ] [MD5:04029d366c97c918521033844574b953,1c9646d7c702dd695f5aac11faa26508,b77376ec90121d252b1e5a46f676fa49]
    POC   16 TId: 0 ( I-SLICE, nQP 21 QP 21 )     750240 bits [Y 41.8177 dB    U 45.0618 dB    V 45.4064 dB] [ET     4 ] [L0 ] [L1 ] [MD5:08bbc78cb251ca53880afab6e45af93e,ed9852c6fe39148b039afc75a621f8ad,8ab4a34720661a0fbe07b0d2a0f2c101]
    POC   17 TId: 0 ( I-SLICE, nQP 21 QP 21 )     522688 bits [Y 42.5521 dB    U 47.0223 dB    V 47.6010 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4bb9f695c253b8eba59e9c4cd4653d58,6e63e60feba09cf7933750a3cb213f29,fd14fe4185a01c2bbd99cd571f14926d]
    POC   18 TId: 0 ( I-SLICE, nQP 21 QP 21 )     590896 bits [Y 42.4926 dB    U 45.4630 dB    V 45.9616 dB] [ET     4 ] [L0 ] [L1 ] [MD5:32d0e59d140d82688b7858d81bdbc7ce,e558c4f8407923abe172261198618260,bcded9d896a43b742d4c27435e070ae3]
    POC   19 TId: 0 ( I-SLICE, nQP 21 QP 21 )     241936 bits [Y 43.9936 dB    U 47.7329 dB    V 47.7361 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a606b9cfa3d22f2fb559ea8bfa4e6410,2490d779973f545f2f888dd765a4986b,1b3ed068b2b559adb01ccd1907b06253]
    POC   20 TId: 0 ( I-SLICE, nQP 21 QP 21 )     284320 bits [Y 42.7487 dB    U 48.3448 dB    V 49.0780 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7840b751bb95e0e604656318e8648703,2a78c4aef0d9362f08d8f5d4d007b241,718c94d26bb820eb07f73542a943df5e]
    POC   21 TId: 0 ( I-SLICE, nQP 21 QP 21 )     563936 bits [Y 42.7259 dB    U 46.2959 dB    V 48.1114 dB] [ET     6 ] [L0 ] [L1 ] [MD5:cec19cce903e1c13dca5776248856c1f,490b42c63dba0f253528740aa9f00062,7c697f66a21ee80d8df7c1f86c332570]
    POC   22 TId: 0 ( I-SLICE, nQP 21 QP 21 )     336696 bits [Y 44.6225 dB    U 47.4689 dB    V 48.9052 dB] [ET     6 ] [L0 ] [L1 ] [MD5:edc9fa12322c979e2408e17b4df56890,5d8b0aa1a7ffeeaee6f5e0e1083215d9,7849ed67458f3a97edfd6cde0af2c99f]
    POC   23 TId: 0 ( I-SLICE, nQP 21 QP 21 )     855688 bits [Y 41.9328 dB    U 45.6942 dB    V 45.9166 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9f693a6ab46810aadec0f8459da02046,055396d12b4f4c1e4ad7beea6557878e,a9cc0c488e4356d66e5c925b73fa2316]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   32819.4200   42.9666   47.1027   47.2998   43.8973  

# 20
    POC    0 TId: 0 ( I-SLICE, nQP 20 QP 20 )     634864 bits [Y 43.6078 dB    U 48.2875 dB    V 46.9542 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4f62996887a97ce3de53b91c48e6abfc,e70bee2a483f35cc78eac4b0e3db4f35,c269e422a5094aaf055d6f25ffd6587a]
    POC    1 TId: 0 ( I-SLICE, nQP 20 QP 20 )     896056 bits [Y 43.4045 dB    U 45.7788 dB    V 46.1939 dB] [ET     4 ] [L0 ] [L1 ] [MD5:6becf27f29433f4d06e9782f8cbf3c7b,850b59a5b67b4242bc1749621ce32c9a,708b93c1c86c5fa47ef484483c33ad79]
    POC    2 TId: 0 ( I-SLICE, nQP 20 QP 20 )     487304 bits [Y 43.5225 dB    U 49.2184 dB    V 46.4114 dB] [ET     4 ] [L0 ] [L1 ] [MD5:bfdc9199efa34514d9be254728fe779e,48d70c86544afeafd38a3f552d91da98,ad31362b117d3b0c2526e514d5db3b5b]
    POC    3 TId: 0 ( I-SLICE, nQP 20 QP 20 )     359296 bits [Y 43.2790 dB    U 48.2313 dB    V 48.6844 dB] [ET     4 ] [L0 ] [L1 ] [MD5:bcff76e0c4e3e5fafb281ace070c6281,2c2edbd05c8d5e44d3a9c9d12290810a,f47497d4f12a999250a60505478d7817]
    POC    4 TId: 0 ( I-SLICE, nQP 20 QP 20 )     703384 bits [Y 43.6239 dB    U 47.4013 dB    V 48.1130 dB] [ET     4 ] [L0 ] [L1 ] [MD5:fd3ddd8d8ba095332084a88cf0fe873a,a360e1ed128c392369a62637c881ff7c,041565fbdb39642c6379e447599a3b84]
    POC    5 TId: 0 ( I-SLICE, nQP 20 QP 20 )     379680 bits [Y 43.9731 dB    U 49.5202 dB    V 49.1683 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2991248d30321d72b9a86d4c155fd320,d4f66b3f05731c4e47879205fa81ee46,63317ec24795fbf47308a9660f8cfb71]
    POC    6 TId: 0 ( I-SLICE, nQP 20 QP 20 )    1260832 bits [Y 43.3270 dB    U 45.0200 dB    V 46.7146 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e40cfa41b1f06267d310814fae6255f2,a9d9a105c81a46b8afa5b85afa6285c0,381cca7560bf508e0db3c320594f4664]
    POC    7 TId: 0 ( I-SLICE, nQP 20 QP 20 )     366472 bits [Y 45.0343 dB    U 47.8365 dB    V 48.4858 dB] [ET     3 ] [L0 ] [L1 ] [MD5:043b6b10c128a75af105bf0cb78941af,bad87479eec08c500ae1bd1e106f5178,8ca3012cad70e83337e22dfaeeafa94a]
    POC    8 TId: 0 ( I-SLICE, nQP 20 QP 20 )     306480 bits [Y 45.1176 dB    U 49.0667 dB    V 49.6484 dB] [ET     3 ] [L0 ] [L1 ] [MD5:ca85518bd12a017655a4ad7e1897c37c,9aa9bb842cc84e420b5730e53ccb98bf,7845156c1132451e70e9d105e54445c4]
    POC    9 TId: 0 ( I-SLICE, nQP 20 QP 20 )     449248 bits [Y 43.6707 dB    U 48.2909 dB    V 48.8967 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7072af1bfa331b53ac3c72b1880e47a2,f36b5b7bb0bbacae3aaf561a26ad5511,7b732a2b2f1a0145b198c21eefc0a990]
    POC   10 TId: 0 ( I-SLICE, nQP 20 QP 20 )     504152 bits [Y 43.9907 dB    U 48.5246 dB    V 49.5506 dB] [ET     3 ] [L0 ] [L1 ] [MD5:10795bf0cc3b35f3a91b8fdb1f1f6705,5ea85a9968e12c58e51b19120ae605e6,cc087ae3962890cb780b6a7ca8591565]
    POC   11 TId: 0 ( I-SLICE, nQP 20 QP 20 )     481592 bits [Y 43.5724 dB    U 47.9967 dB    V 46.0434 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f7663019656022969bc323364b82959b,1b6c72f6090500406146651211ead664,2539c41e0d8168fc0fd2794c2f53aeb8]
    POC   12 TId: 0 ( I-SLICE, nQP 20 QP 20 )     828080 bits [Y 43.1919 dB    U 46.7663 dB    V 47.0853 dB] [ET     4 ] [L0 ] [L1 ] [MD5:cd75bf8a53481787f2ea0450123a9dee,e380ca0ca892fcb7d8dd8ed9dfcf6f0f,73f05ab999e5418d1825110b4975617e]
    POC   13 TId: 0 ( I-SLICE, nQP 20 QP 20 )     943064 bits [Y 43.3215 dB    U 47.9810 dB    V 47.4619 dB] [ET     4 ] [L0 ] [L1 ] [MD5:cbe9343a7ad360f45eda1e298244bdd7,0cced36f2a55b63dcc683fa98067c947,e9b844bdd156197c45e00ac1bf668732]
    POC   14 TId: 0 ( I-SLICE, nQP 20 QP 20 )     436456 bits [Y 44.1417 dB    U 48.6120 dB    V 47.3886 dB] [ET     3 ] [L0 ] [L1 ] [MD5:11f11d645390d6b4e9749c1303f1a6bc,ceb461c5e9d597b3cd6d7e8321c8ba84,53a22241b2c1156fa4dacfd6aa1bd2fb]
    POC   15 TId: 0 ( I-SLICE, nQP 20 QP 20 )     754440 bits [Y 43.8622 dB    U 46.2930 dB    V 47.1618 dB] [ET     4 ] [L0 ] [L1 ] [MD5:60bf2af040b85b266288fd4d298bdcc0,d051abbe53212d078a138ef3c2253c81,aa2a3cd3085c491e2386c51b2576365f]
    POC   16 TId: 0 ( I-SLICE, nQP 20 QP 20 )     824104 bits [Y 42.6525 dB    U 45.6395 dB    V 45.9682 dB] [ET     4 ] [L0 ] [L1 ] [MD5:2dd3ba9886d753eddf7f0e029cfd84ed,b6637867e60bb64952ba0d7782788ad6,fc79384e3327f1bcff04e02303286799]
    POC   17 TId: 0 ( I-SLICE, nQP 20 QP 20 )     570640 bits [Y 43.2405 dB    U 47.4718 dB    V 47.9948 dB] [ET     3 ] [L0 ] [L1 ] [MD5:4494a51e10255ba9e91f6d0b01c90126,24ba2b30374760862b93c7663b853b6b,d467964fa67d57216728d7030630c87f]
    POC   18 TId: 0 ( I-SLICE, nQP 20 QP 20 )     648400 bits [Y 43.2437 dB    U 46.0168 dB    V 46.5455 dB] [ET     4 ] [L0 ] [L1 ] [MD5:04237dfc6e5fd34efabbeb4601c0fe93,c303c6aa99415b8d2af5255a572d6b74,7623ca90382ee84b79db5d85893a6576]
    POC   19 TId: 0 ( I-SLICE, nQP 20 QP 20 )     269256 bits [Y 44.4333 dB    U 48.3022 dB    V 48.2364 dB] [ET     3 ] [L0 ] [L1 ] [MD5:d504b38624df8d272143910b99aec318,1f11617cd0d6d13ce5b8c6f51440e9f0,6427296e69bc0d517f27b89c2a299c4c]
    POC   20 TId: 0 ( I-SLICE, nQP 20 QP 20 )     314616 bits [Y 43.1723 dB    U 48.7545 dB    V 49.4726 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e178b829092ff605213129f46cb46518,1d5904c03ce6a42e131872b40aa81c87,f7b16cd27609b659dda417f1b4385301]
    POC   21 TId: 0 ( I-SLICE, nQP 20 QP 20 )     609408 bits [Y 43.3639 dB    U 46.7664 dB    V 48.4909 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d2b2556d9e2445916d365dfc292d268e,cdeac1acfdc9ef55025e2f1dc39a3a3c,a27a4888d8bfda3e2f62b1046f6355e8]
    POC   22 TId: 0 ( I-SLICE, nQP 20 QP 20 )     369536 bits [Y 45.3571 dB    U 47.8600 dB    V 49.3825 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0e5a54dd1849e3736cb314f54f015a5b,ab6e2ce131c040a4ccd71f088ec410ed,6c9ffccddd0cc344c0bbfe108c5000f4]
    POC   23 TId: 0 ( I-SLICE, nQP 20 QP 20 )     928608 bits [Y 42.8454 dB    U 46.1840 dB    V 46.4319 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b59b0e9f8ccc902878d6cd8d4ea44bbb,a68a07cfbcc1f96dad1990b32ca7172d,a1007c19ff5f8df79ec2344f7c472349]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   35814.9200   43.7062   47.5758   47.7702   44.6030  

# 19
    POC    0 TId: 0 ( I-SLICE, nQP 19 QP 19 )     696592 bits [Y 44.5065 dB    U 48.8367 dB    V 47.4879 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1e7168d1ff152755ce5931a82c0250fe,47c6aa5811ced33ad5c3f380636cf6de,0e01a46612eb0407eec63f7ca0ba7222]
    POC    1 TId: 0 ( I-SLICE, nQP 19 QP 19 )     967192 bits [Y 44.3541 dB    U 46.2989 dB    V 46.7562 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3a089d64151780db8990b44302b2c27f,91df8519262f37032d2b2379cefa99a2,8f1c30c53a863d7d53b225c84f55967a]
    POC    2 TId: 0 ( I-SLICE, nQP 19 QP 19 )     547832 bits [Y 44.3679 dB    U 49.6371 dB    V 46.9414 dB] [ET     4 ] [L0 ] [L1 ] [MD5:054ab588d0ec3f56da62333919da4f29,38b27a82e612464dd01f01416830fde8,424927cda9bae3e2139f70162cd54d74]
    POC    3 TId: 0 ( I-SLICE, nQP 19 QP 19 )     410776 bits [Y 43.9191 dB    U 48.6482 dB    V 49.1529 dB] [ET     4 ] [L0 ] [L1 ] [MD5:871cab047dc5295f27a035771c52f38c,096235bcec998e7cc2360973fbf50f17,260ce159f2df6f1e11cc13051819dcd1]
    POC    4 TId: 0 ( I-SLICE, nQP 19 QP 19 )     767960 bits [Y 44.5765 dB    U 47.9500 dB    V 48.6537 dB] [ET     4 ] [L0 ] [L1 ] [MD5:95536c423d7219b452f22ebf25e92739,367975ad6fd4534517a9ee8489749ecb,009e876e4288803bc7745663176807ed]
    POC    5 TId: 0 ( I-SLICE, nQP 19 QP 19 )     426456 bits [Y 44.7086 dB    U 49.8625 dB    V 49.5854 dB] [ET     3 ] [L0 ] [L1 ] [MD5:f55469b9f37a81d17e6e2adb715b75f6,70a06f7095cd28b6d924c479cd176c62,30127119383c1ed63ec2f95c66211e6e]
    POC    6 TId: 0 ( I-SLICE, nQP 19 QP 19 )    1343888 bits [Y 44.3748 dB    U 45.5873 dB    V 47.1862 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e6e8d5dcefb1d1c7e9e5911df73735a7,6b17c71232a9f5437c4cac3b3e1f9500,e3051557efbe4057057db9ce17926d28]
    POC    7 TId: 0 ( I-SLICE, nQP 19 QP 19 )     402080 bits [Y 45.6621 dB    U 48.3640 dB    V 49.0255 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3f5478d038149d67348639385d5b00b0,fbdd5bb419b4d7a72fd31050e490d83b,f0220a6c6129933d68034359bcbde8f4]
    POC    8 TId: 0 ( I-SLICE, nQP 19 QP 19 )     339840 bits [Y 45.7713 dB    U 49.6012 dB    V 50.0580 dB] [ET     3 ] [L0 ] [L1 ] [MD5:c4a3b48923ad2ff7d5f8683c9304cf46,c0b6f2ce516c2332bf0c3cc52991faff,2b3a4c8c6e67e624633cb5c9c7101486]
    POC    9 TId: 0 ( I-SLICE, nQP 19 QP 19 )     499408 bits [Y 44.3906 dB    U 48.7071 dB    V 49.2203 dB] [ET     3 ] [L0 ] [L1 ] [MD5:7468306f40f4a53429cad7afba08f367,ebbca8f41342dec80974d27b6bd1cacf,c4e5c15ab02bf6c9abacbc236c6debcd]
    POC   10 TId: 0 ( I-SLICE, nQP 19 QP 19 )     557032 bits [Y 44.8386 dB    U 48.9395 dB    V 49.9089 dB] [ET     3 ] [L0 ] [L1 ] [MD5:cac80795d13130510dd5c7ca53afc226,138aa1c3a99340b6a862b0300f3cd04e,bddda8b3d42c710080c658b86f182453]
    POC   11 TId: 0 ( I-SLICE, nQP 19 QP 19 )     542888 bits [Y 44.4144 dB    U 48.5088 dB    V 46.6212 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9154aef6179d1b73a4b3e7485c628b71,7db9dc43bbf8a7a88e16cbf39766edcf,b28a3b49355a7b2a4e1b9177c8664b92]
    POC   12 TId: 0 ( I-SLICE, nQP 19 QP 19 )     901608 bits [Y 44.1590 dB    U 47.3712 dB    V 47.6267 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7cedaabb053816fafd15ac21c263f0ac,c8ddd24b6dbd1500d390e2b91995431a,d57d10f37d7df95b916a531acb24be92]
    POC   13 TId: 0 ( I-SLICE, nQP 19 QP 19 )    1013336 bits [Y 44.3197 dB    U 48.4831 dB    V 47.9615 dB] [ET     4 ] [L0 ] [L1 ] [MD5:02c709041920ce2c8157d37c5d148d31,1aac2042eeaeb4add5575d853b7c14d3,dc06d7cd1fdc9a684a071e9b404265d2]
    POC   14 TId: 0 ( I-SLICE, nQP 19 QP 19 )     485448 bits [Y 44.9388 dB    U 49.0626 dB    V 47.9921 dB] [ET     3 ] [L0 ] [L1 ] [MD5:0d1f7d801b8d6dec570c5f6891c504b0,d000e68ff0350c21fd81cee0b8976ea8,7d5ce24268319e2a8057cf4bb2d2d5ce]
    POC   15 TId: 0 ( I-SLICE, nQP 19 QP 19 )     816216 bits [Y 44.7341 dB    U 46.9669 dB    V 47.7599 dB] [ET     4 ] [L0 ] [L1 ] [MD5:bb7e5b679d8503dc3697ebf3e7a19d01,fb5040cfd29a84a227a0a9c4bd6f4f3f,0cc90b63637d1c5058f6ca59857433eb]
    POC   16 TId: 0 ( I-SLICE, nQP 19 QP 19 )     912712 bits [Y 43.6501 dB    U 46.2310 dB    V 46.5501 dB] [ET     6 ] [L0 ] [L1 ] [MD5:50d7edd0825c663cf6a208dd4abcfd53,3838bb4e769cfd2d7fc24070fdf3f4a6,db0b4dcbd8dd45b0717d9de23b1d781f]
    POC   17 TId: 0 ( I-SLICE, nQP 19 QP 19 )     628168 bits [Y 43.9999 dB    U 47.8719 dB    V 48.4039 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d30b70222a0a6850c4b012d9650ca0f5,98a5b66625ecec37d874a6e3847ff765,5a26137a845c8a1f0b91412b29825ca5]
    POC   18 TId: 0 ( I-SLICE, nQP 19 QP 19 )     714240 bits [Y 44.0591 dB    U 46.5595 dB    V 47.0749 dB] [ET     4 ] [L0 ] [L1 ] [MD5:068d2e79e79a7bde106ae178b5183341,4b9cc55ee3255c979bf860fd129e1d18,8f0fab2330c1ff23b345cdf0223363c9]
    POC   19 TId: 0 ( I-SLICE, nQP 19 QP 19 )     305160 bits [Y 44.9749 dB    U 48.6944 dB    V 48.7170 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1b40bca679e3ef94374bc688798fa9e3,c155f2456dfa9efea5517854bd934ca3,20cc85bb9c821fd1058d440b7a31ed4b]
    POC   20 TId: 0 ( I-SLICE, nQP 19 QP 19 )     358168 bits [Y 43.6928 dB    U 49.1664 dB    V 49.8523 dB] [ET     3 ] [L0 ] [L1 ] [MD5:20b3576cfb3aa8e37389aeeb59e3dbe9,0a402991fbf1c53cd97df332cd157fce,9ba24de3870112f87dd93f76cda13c09]
    POC   21 TId: 0 ( I-SLICE, nQP 19 QP 19 )     663080 bits [Y 44.0473 dB    U 47.2132 dB    V 48.8877 dB] [ET     3 ] [L0 ] [L1 ] [MD5:70af8a0c0e40cf32786e34b916d3638b,7078839a534f244e9bfce299d2b7587d,e9123c0cac4d3a1034b8b56152c5edcd]
    POC   22 TId: 0 ( I-SLICE, nQP 19 QP 19 )     407392 bits [Y 46.1495 dB    U 48.2204 dB    V 49.8138 dB] [ET     3 ] [L0 ] [L1 ] [MD5:1d6580e8ecc5e6a8ab15eb6b9465fa49,9b269370b4409a1769626e50561cd521,4a5ade6dda7fcc9e48ca8cc39f9516ba]
    POC   23 TId: 0 ( I-SLICE, nQP 19 QP 19 )    1007960 bits [Y 43.8527 dB    U 46.7242 dB    V 46.9787 dB] [ET     4 ] [L0 ] [L1 ] [MD5:06d9b46c041358959690d8a43143ca70,f5b0b1f821c86cbf873e63e5e7e011eb,c5e8901b10ea21989fa66b16bec59356]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   39288.5800   44.5193   48.0628   48.2590   45.3674  

# 18
    POC    0 TId: 0 ( I-SLICE, nQP 18 QP 18 )     760504 bits [Y 45.3865 dB    U 49.2731 dB    V 48.0478 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ed13ce766428a1007cfdf6b01a396add,0a186bb6a0a79f4ffa0b2900f94a215c,e805937fdf351ee096283a22a847aa36]
    POC    1 TId: 0 ( I-SLICE, nQP 18 QP 18 )    1040544 bits [Y 45.2599 dB    U 46.9544 dB    V 47.4450 dB] [ET     5 ] [L0 ] [L1 ] [MD5:888a0a37e3a4ddf8c530e355b3cf7928,0e52c7e8fb79ffe06ea58dd16cd1ff94,92a10d6c554a8448fd38af4515472782]
    POC    2 TId: 0 ( I-SLICE, nQP 18 QP 18 )     612488 bits [Y 45.2195 dB    U 50.0363 dB    V 47.5911 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5d62f09b58e64fb229863ddfdb401a64,e89124545e16e3804c64cf8362c0a1e4,f3c1fe0c40c6b7a25229023b2c517ee0]
    POC    3 TId: 0 ( I-SLICE, nQP 18 QP 18 )     472368 bits [Y 44.6286 dB    U 49.1709 dB    V 49.6101 dB] [ET     4 ] [L0 ] [L1 ] [MD5:a3466c9483051bf2c505f0096883ad68,db671efe1fb1d65f352b0910d620cadc,d984f4041a48668dac20940f38a1a500]
    POC    4 TId: 0 ( I-SLICE, nQP 18 QP 18 )     831368 bits [Y 45.4898 dB    U 48.3971 dB    V 49.1216 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e75c7f4bbb3e9efa254465064f7498e2,35eebc507b48853bc206aa9312740746,ca02f2cc422a614c9527924e01948dc4]
    POC    5 TId: 0 ( I-SLICE, nQP 18 QP 18 )     475512 bits [Y 45.4436 dB    U 50.3543 dB    V 50.0365 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4ed55472b0e3d2066e1946a524961be7,1d4452eabf410d63c99fe0fc59f1a1e7,47ef26532f32e2f56688ec42f0ea1294]
    POC    6 TId: 0 ( I-SLICE, nQP 18 QP 18 )    1425240 bits [Y 45.4193 dB    U 46.2282 dB    V 47.6726 dB] [ET     5 ] [L0 ] [L1 ] [MD5:fdac657f0bf027722d3ed5347f64d86d,90882251fffd6f17b77f95cbf6e49a8e,799fd9ea8f3b58713a2e74f93abd38a8]
    POC    7 TId: 0 ( I-SLICE, nQP 18 QP 18 )     439048 bits [Y 46.2493 dB    U 48.8327 dB    V 49.6045 dB] [ET     3 ] [L0 ] [L1 ] [MD5:6a9c2c5f1451d81c5fca479796e8d955,fe6db16f8fa4dda2518ddfef0be2b063,1e5c1f40ec52495a7a7c8481d1f7791f]
    POC    8 TId: 0 ( I-SLICE, nQP 18 QP 18 )     373496 bits [Y 46.3806 dB    U 49.9285 dB    V 50.5162 dB] [ET     3 ] [L0 ] [L1 ] [MD5:206eeafb3a64b5ff4dd1286bb170e104,deeaf040f3d147b56811c869326ca81b,811247407a95bd07b7b2bfe6d7020f7e]
    POC    9 TId: 0 ( I-SLICE, nQP 18 QP 18 )     552560 bits [Y 45.0937 dB    U 49.1828 dB    V 49.7791 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b4437916ac81db5352d9797090e80dc0,8e4d1b048cc72fdd18ea5a49687ff2db,3262bf916de97b6f38508793c5f8e621]
    POC   10 TId: 0 ( I-SLICE, nQP 18 QP 18 )     608704 bits [Y 45.6374 dB    U 49.3359 dB    V 50.2319 dB] [ET     3 ] [L0 ] [L1 ] [MD5:67059471afd0638a7461e0d736d07100,3fd2ed6b9ee925e1c4e73fb12ed474aa,7d3f6030ad51e07c0a045532b90764a1]
    POC   11 TId: 0 ( I-SLICE, nQP 18 QP 18 )     607928 bits [Y 45.2788 dB    U 49.0470 dB    V 47.2006 dB] [ET     4 ] [L0 ] [L1 ] [MD5:68ae2ac56c4846a4f840ef01f11946c4,e23363123b3f749c729a4d00c734e0d2,b0f09e22d184414154da06883730be5d]
    POC   12 TId: 0 ( I-SLICE, nQP 18 QP 18 )     978456 bits [Y 45.1339 dB    U 47.9424 dB    V 48.2710 dB] [ET     4 ] [L0 ] [L1 ] [MD5:28bfab9854b6b4703abd86e6dbf58f1b,aac305864522a3ba3e681d178d134b12,5953eef1c54d45a92f8436fbf5b504ab]
    POC   13 TId: 0 ( I-SLICE, nQP 18 QP 18 )    1082248 bits [Y 45.2753 dB    U 48.9230 dB    V 48.4641 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e53cdffcc4853470ee773474ecb3c5d0,71e2be3686178ccffa9dc40d89ea095d,af3637831a2f37af2560e2a0197af02e]
    POC   14 TId: 0 ( I-SLICE, nQP 18 QP 18 )     538712 bits [Y 45.7405 dB    U 49.4572 dB    V 48.5748 dB] [ET     3 ] [L0 ] [L1 ] [MD5:faf007a0f167b279a7508ae0881f8038,55344d1d09a0240460e1e4378584a2ab,1f5b10c70dee633203d55e512b726a08]
    POC   15 TId: 0 ( I-SLICE, nQP 18 QP 18 )     880560 bits [Y 45.6389 dB    U 47.5706 dB    V 48.2801 dB] [ET     4 ] [L0 ] [L1 ] [MD5:8b04733206cc88ef83dbd03edbb9539b,ac6f518ed8829f87c5703945861baf8e,a27ff6301b2765febb0d4e05a25714b8]
    POC   16 TId: 0 ( I-SLICE, nQP 18 QP 18 )    1007712 bits [Y 44.7136 dB    U 46.8236 dB    V 47.1473 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d15762861e6825b31575abed56aeceab,a1d1e52c4ac163b04841fac298b037c6,4d39b6c5ccd648e8d8c57648f3bdbaba]
    POC   17 TId: 0 ( I-SLICE, nQP 18 QP 18 )     692624 bits [Y 44.7912 dB    U 48.3141 dB    V 48.8469 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e419919cd294e5ee0b728d7b1711f495,2db4559f9fb35097208857582c9c5df1,5c7febf46ea385033da1d86c28a7948f]
    POC   18 TId: 0 ( I-SLICE, nQP 18 QP 18 )     786888 bits [Y 44.9120 dB    U 47.1249 dB    V 47.6728 dB] [ET     4 ] [L0 ] [L1 ] [MD5:157f2f1483d31e9d8d8b7aa8e27d670e,8adc6ab64dc595157ebe772228974867,05b5c9c3827d01af7b42e742707fce17]
    POC   19 TId: 0 ( I-SLICE, nQP 18 QP 18 )     346072 bits [Y 45.5247 dB    U 49.1757 dB    V 49.1751 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e16bdcae4694ec5c71f33a3a06d23b4c,1b1061c5d77402dfc1078b4d8b279bb9,8e0144f43c0b5f0ca3c5b10ea912bc8e]
    POC   20 TId: 0 ( I-SLICE, nQP 18 QP 18 )     425424 bits [Y 44.4274 dB    U 49.5746 dB    V 50.1937 dB] [ET     4 ] [L0 ] [L1 ] [MD5:35964a1dfe7e642ab9ebe4c5393dfe01,4adb55045953b3cc5dfdf9ef4f337428,209958004e3e3b6ad2fe1a77c1a2a4d8]
    POC   21 TId: 0 ( I-SLICE, nQP 18 QP 18 )     727520 bits [Y 44.7920 dB    U 47.7363 dB    V 49.3181 dB] [ET     4 ] [L0 ] [L1 ] [MD5:0c1409c9c6f91bfce931e931d54caabc,981917dc5a13371b1a667d43a8d6ffc4,83c3cd3873fa7d0c53d13c8e2c184be8]
    POC   22 TId: 0 ( I-SLICE, nQP 18 QP 18 )     447640 bits [Y 46.9441 dB    U 48.5691 dB    V 50.2198 dB] [ET     3 ] [L0 ] [L1 ] [MD5:5074b0683115b84cf61cc7b5674214ce,e766ebd8298b6c4b1400e45b3ed46904,99b5db162adb480d751066e81fcb6038]
    POC   23 TId: 0 ( I-SLICE, nQP 18 QP 18 )    1089904 bits [Y 44.8776 dB    U 47.2248 dB    V 47.5229 dB] [ET     5 ] [L0 ] [L1 ] [MD5:998b14991d89ae54a495f2215324f7b3,fad3e6a088ec0dc4ec23b47544c889a9,a5bdf6c768c48e1c2637b0eed62dfb6c]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   43008.8000   45.3441   48.5491   48.7727   46.1403  

# 17
    POC    0 TId: 0 ( I-SLICE, nQP 17 QP 17 )     824536 bits [Y 46.2353 dB    U 49.6861 dB    V 48.4808 dB] [ET     5 ] [L0 ] [L1 ] [MD5:46210f0b22184fd76b43d0a3528a5ed5,c7c2bc14c840d9be23774bb828027646,ab0dabae1c998db77cbe4fa6c4623e5a]
    POC    1 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1108264 bits [Y 46.0846 dB    U 47.4594 dB    V 47.9417 dB] [ET     5 ] [L0 ] [L1 ] [MD5:546f872b056718ff4efdf2a466e3cb49,6df9f62280da4c8f59b8be50f941f092,ffb766284f7fe4fa607d37b1bb893cd7]
    POC    2 TId: 0 ( I-SLICE, nQP 17 QP 17 )     676128 bits [Y 46.0278 dB    U 50.2990 dB    V 48.1765 dB] [ET     4 ] [L0 ] [L1 ] [MD5:a1cbd39a99990ec2fa05a7bce0bb4cf8,873820e76c1be55353e09ed1903608c3,3e31e06be2eaa0fad80faa50d8e04159]
    POC    3 TId: 0 ( I-SLICE, nQP 17 QP 17 )     544224 bits [Y 45.4523 dB    U 49.5364 dB    V 49.8891 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5dd973a5222032d68c13b41c728c5d7c,2a388c3b8f2e9be1c9a6a8d855dab47c,9159d73db226da34aa35fd854e1aa9e4]
    POC    4 TId: 0 ( I-SLICE, nQP 17 QP 17 )     893792 bits [Y 46.3352 dB    U 48.8766 dB    V 49.5461 dB] [ET     4 ] [L0 ] [L1 ] [MD5:97c580b79f388e63f72a5b881d19d16a,34d93588bcb2466ad0024c2c6f9e6eea,3fe650be942a224aae705b0976b4da36]
    POC    5 TId: 0 ( I-SLICE, nQP 17 QP 17 )     529512 bits [Y 46.1834 dB    U 50.7594 dB    V 50.4333 dB] [ET     4 ] [L0 ] [L1 ] [MD5:bcb28c77c9e53363fb2b35c9e999320d,d78a6758e0ff396cd62e794883c0de84,49d564a3f3cabf4804153ca10bcf9fdc]
    POC    6 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1502408 bits [Y 46.3592 dB    U 46.7703 dB    V 48.0815 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3c22be385cda2c9c47f85a17b9d9c38c,8ce85b27c70157f2e27bab6ef740c81e,94286e8f29468fbbe305e543fc3718e2]
    POC    7 TId: 0 ( I-SLICE, nQP 17 QP 17 )     474048 bits [Y 46.7620 dB    U 49.3178 dB    V 49.9650 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7e9619e4925e87f1dbab125947f79ff4,605bb10b33a34b079652f083672b23ab,af86688f0cbd78288c3f8b362a9b05d8]
    POC    8 TId: 0 ( I-SLICE, nQP 17 QP 17 )     410320 bits [Y 46.9572 dB    U 50.4466 dB    V 50.9004 dB] [ET     3 ] [L0 ] [L1 ] [MD5:2685239ccd81d15c4af4201fa4d6fa9a,738f08221fc3eed9c988e93cd874ab5c,f61717ba03c2bc725f5932f2dacd02b7]
    POC    9 TId: 0 ( I-SLICE, nQP 17 QP 17 )     612144 bits [Y 45.8369 dB    U 49.5759 dB    V 50.0766 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1879d0f63686d78a576bba985ef41992,61451ec4183d6f1af690222d47a361a3,4336305edfa3653ee1a1ac115ee751d2]
    POC   10 TId: 0 ( I-SLICE, nQP 17 QP 17 )     661344 bits [Y 46.4081 dB    U 49.7434 dB    V 50.5816 dB] [ET     4 ] [L0 ] [L1 ] [MD5:720705eca60fe71f6bf4ed97d1bb789f,187630d5c18e87bfbb115ce4141432f9,5b8af89a5f0abe4d38ea1a07323dd292]
    POC   11 TId: 0 ( I-SLICE, nQP 17 QP 17 )     672336 bits [Y 46.0958 dB    U 49.4745 dB    V 47.6563 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f3999f3db127ea9adf7e357b1d7bb05c,967553121180fb6fd92d0e1c31614ee6,84a84cc6c8a91fe2b1bf721bd47eaaff]
    POC   12 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1052776 bits [Y 46.0665 dB    U 48.4619 dB    V 48.7690 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1406f3295654509b159191c1e42519de,e338a4756a1c695e67b29283602d84bd,4628086a6a387c5fd68bed4546c0c85e]
    POC   13 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1150280 bits [Y 46.1843 dB    U 49.3671 dB    V 48.9230 dB] [ET     5 ] [L0 ] [L1 ] [MD5:75df544d6a41a0e358fb7ccdc47d7d20,fb4f469477dbdda01a281c5ae056134a,80f3a72156fd221a2e3b64a92c95c802]
    POC   14 TId: 0 ( I-SLICE, nQP 17 QP 17 )     594232 bits [Y 46.4989 dB    U 49.9331 dB    V 49.0572 dB] [ET     4 ] [L0 ] [L1 ] [MD5:609f557bd5cfc8cece461ed387e2cd93,31a48c3454021546ddc29eed0936c8fb,369b95d61d6a13eb147216f2edbe83ad]
    POC   15 TId: 0 ( I-SLICE, nQP 17 QP 17 )     942768 bits [Y 46.4440 dB    U 48.1516 dB    V 48.7675 dB] [ET     4 ] [L0 ] [L1 ] [MD5:8604b58f5f86fb19df0ea883fb8f59fd,ff717df814d8e1cabf78ef7ed6c1b453,17eca0cc02df291685291e3ba8c14f4b]
    POC   16 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1096696 bits [Y 45.7282 dB    U 47.3515 dB    V 47.6771 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c694eeab98e944edf31315e3df33be9a,f15da5c45477c9a9e51f9c7cb7a5da75,2b07365700c2e4d53dda310050404208]
    POC   17 TId: 0 ( I-SLICE, nQP 17 QP 17 )     760376 bits [Y 45.5729 dB    U 48.6978 dB    V 49.2287 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d1dc6df291e4d38306cf98ec56ce2086,c7b4833fd71ab1fcce80f8cffdb5e607,9697607f2fe14334869cfcbba20389e1]
    POC   18 TId: 0 ( I-SLICE, nQP 17 QP 17 )     861768 bits [Y 45.7554 dB    U 47.5930 dB    V 48.1944 dB] [ET     4 ] [L0 ] [L1 ] [MD5:6934fa9aca1b1b346092c967cc5d320c,092f46dc87ee21565db6a590017d764d,ed5c8c0f2ab8e48f6cf1e9f70dadd80e]
    POC   19 TId: 0 ( I-SLICE, nQP 17 QP 17 )     390088 bits [Y 46.0776 dB    U 49.5521 dB    V 49.6023 dB] [ET     3 ] [L0 ] [L1 ] [MD5:8cdb9c99f625f8f4ee3f6169f266981c,7f330f7ec620d91abc3b3d41b9c04133,6a5b2a4acf96482b0f2603010a1d8ff8]
    POC   20 TId: 0 ( I-SLICE, nQP 17 QP 17 )     508704 bits [Y 45.3498 dB    U 49.9533 dB    V 50.5770 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1b5a642d5ec15bb2ab566b5a83948b33,7ed9937b0bbae700514a6cca451afe2e,27915f5121f70059dbc8f6d27f82307c]
    POC   21 TId: 0 ( I-SLICE, nQP 17 QP 17 )     803024 bits [Y 45.6183 dB    U 48.1890 dB    V 49.6649 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d3bf7aff41976603801a7214921bf4ff,98b49063ce63a8f0c3424acd880425cb,d720ff26e17b71fc1e56dc4efdcfaeec]
    POC   22 TId: 0 ( I-SLICE, nQP 17 QP 17 )     490360 bits [Y 47.7251 dB    U 48.8871 dB    V 50.5946 dB] [ET     3 ] [L0 ] [L1 ] [MD5:e932256c3f842d745f428cfec7fb0024,596e6bd54eeb83cf263d49b66e042f0a,bc231acfee57d4b49f6a5071cee27601]
    POC   23 TId: 0 ( I-SLICE, nQP 17 QP 17 )    1167856 bits [Y 45.8005 dB    U 47.7497 dB    V 47.9914 dB] [ET     4 ] [L0 ] [L1 ] [MD5:fe555773ea870541c3d1786b322d606c,00270605d52c53f7d47382b09ef7707d,84e23f741989eb744e264f84ac67d909]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   46819.9600   46.1483   48.9930   49.1990   46.8793  

# 16
    POC    0 TId: 0 ( I-SLICE, nQP 16 QP 16 )     894912 bits [Y 47.1494 dB    U 50.1271 dB    V 49.0248 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a35418f0a9cc7b6522f5ad3eec1bbc55,87d6c1a3320ff2aff1b36edaac19ed49,85a81ba051913cffb22414a6eada000e]
    POC    1 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1187640 bits [Y 47.0143 dB    U 48.0532 dB    V 48.4817 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3e04a52197ae569c947ca540cef16e48,524d3a1dba3d83599b5dbf960d83d071,cf1348bb8a8ec2b8f14ddaa0336a119e]
    POC    2 TId: 0 ( I-SLICE, nQP 16 QP 16 )     751408 bits [Y 46.9495 dB    U 50.7073 dB    V 48.7459 dB] [ET     4 ] [L0 ] [L1 ] [MD5:44b4c04ca4ed03209f0c64481b0b2216,b85e61a50c2bf7036e6854be52fbf60f,733e107160a0b1c8fd9d1174f77e274d]
    POC    3 TId: 0 ( I-SLICE, nQP 16 QP 16 )     633312 bits [Y 46.4759 dB    U 49.9724 dB    V 50.2930 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ea1594b9ddfc5a503e8546391cf59acc,21597fdac4c1af48001b9975bb2097c7,9c6da172156a7e3b9718344283d02078]
    POC    4 TId: 0 ( I-SLICE, nQP 16 QP 16 )     963064 bits [Y 47.2358 dB    U 49.3452 dB    V 50.0236 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b8867fd60b8c21b2775564a42a3eea31,527bfde4366bfa854f5ea25cb0d4b92a,e3ce23d10e5004d8d860b0fdb956436e]
    POC    5 TId: 0 ( I-SLICE, nQP 16 QP 16 )     590448 bits [Y 46.9736 dB    U 51.1198 dB    V 50.8805 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b1722fc1ed46eb535c4a21757e1d1036,fab2954747d929fad1a0b98fd943bc70,4df380cb981dd17e97b2df1afa8db39d]
    POC    6 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1589904 bits [Y 47.3929 dB    U 47.4074 dB    V 48.6192 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ed35bbf02f61b54b67f1722674389b84,194a4bceb90ff6af55e720e8f22812e2,c49728cfbaa1b39e566bd74e2dc04d32]
    POC    7 TId: 0 ( I-SLICE, nQP 16 QP 16 )     520608 bits [Y 47.3719 dB    U 49.8666 dB    V 50.4994 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ee5b4196ccfe8d841728e9ab3711b14a,ebefb8cc3bbc318d697d1fbbbda480de,8a7bc29963396e38d323c3b86b07cd5b]
    POC    8 TId: 0 ( I-SLICE, nQP 16 QP 16 )     451736 bits [Y 47.5660 dB    U 50.8211 dB    V 51.2985 dB] [ET     4 ] [L0 ] [L1 ] [MD5:839bc9b425bc4919a8e2b23e9319bf66,13709164fb58993408102af11b204ed4,e1d6d8c15e804a6e67dca5ee8f1b58bc]
    POC    9 TId: 0 ( I-SLICE, nQP 16 QP 16 )     683040 bits [Y 46.6732 dB    U 49.9747 dB    V 50.4749 dB] [ET     4 ] [L0 ] [L1 ] [MD5:09a1eed3cc64458fbebfca384d63b0a0,495368023b9446637ff9223f0187e6a2,7604c66bf8d256305f13ecada0613d15]
    POC   10 TId: 0 ( I-SLICE, nQP 16 QP 16 )     722240 bits [Y 47.2294 dB    U 50.1525 dB    V 50.9533 dB] [ET     4 ] [L0 ] [L1 ] [MD5:178bfc9cb85cfc2d2dbfbf56ae7c22f6,658ea93bafcd17e034d13fafbc8060ce,f6d374b869dd0098455de8a21888d74b]
    POC   11 TId: 0 ( I-SLICE, nQP 16 QP 16 )     747824 bits [Y 47.0050 dB    U 49.9205 dB    V 48.2460 dB] [ET     4 ] [L0 ] [L1 ] [MD5:fc65bdb193cd78c47fec5f7e282950ff,5dfa197664e669a941f61c6e9bae3014,438fd3213ee7adb9a0b6b2f75d42da6a]
    POC   12 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1132416 bits [Y 47.0363 dB    U 49.0146 dB    V 49.2684 dB] [ET     5 ] [L0 ] [L1 ] [MD5:883bc4374fe1ea2cb0ec6d064b0846bf,d6b9880f9c8449343084a7c84ca00dd9,a852a30e67c18fea219c9e0f914cf395]
    POC   13 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1224536 bits [Y 47.1606 dB    U 49.7576 dB    V 49.3713 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a5e1adfed617870ac0f4815a16b58aaf,d6f9b0c2ee04ca892aae8173b7801a9c,6f612a7da991460c00d6510677ef894d]
    POC   14 TId: 0 ( I-SLICE, nQP 16 QP 16 )     653784 bits [Y 47.2796 dB    U 50.3520 dB    V 49.6003 dB] [ET     4 ] [L0 ] [L1 ] [MD5:8e85b44a0c6164902cdca9ff62481786,21b137c0b7df33b358b133dd8e85aa25,ed7b401632b4b4bac4801ac50f976473]
    POC   15 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1012224 bits [Y 47.3236 dB    U 48.7686 dB    V 49.2848 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0595a25e431a5df5e26cd23a758f000b,d8d86bc8a6a99ff2f9d047ed332f6a6f,a05199e6543517446dceb8a23458f062]
    POC   16 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1187336 bits [Y 46.7960 dB    U 47.9349 dB    V 48.2279 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7222f1cf1a87f99033f59017b0255491,997218c021590572d671282004422341,2f78411e7c72c7b85a6307f1a1cb974c]
    POC   17 TId: 0 ( I-SLICE, nQP 16 QP 16 )     846080 bits [Y 46.5389 dB    U 49.1141 dB    V 49.6218 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9c8bdb719e70c14bef0f681cae24bf20,8e67070954d3176de628d2fb4ed53059,f22cf1f461f8076035128212b515c9c8]
    POC   18 TId: 0 ( I-SLICE, nQP 16 QP 16 )     948872 bits [Y 46.7314 dB    U 48.1733 dB    V 48.7093 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b5d7e60b98200a308ca6a72b7a90f79f,d61f9fc77c37a0a388b24d981326139b,2a187b76791f57afddc1f2c4b667c9b6]
    POC   19 TId: 0 ( I-SLICE, nQP 16 QP 16 )     449640 bits [Y 46.7573 dB    U 49.9162 dB    V 50.0718 dB] [ET     3 ] [L0 ] [L1 ] [MD5:363995ddbade6cd96e48c0cde9120c9a,9938f7c6318f19e5176b126127b1667e,b90aeeb158ea769ba85d14e212ad905f]
    POC   20 TId: 0 ( I-SLICE, nQP 16 QP 16 )     605312 bits [Y 46.4834 dB    U 50.3330 dB    V 50.9193 dB] [ET     4 ] [L0 ] [L1 ] [MD5:de915644c6f636099bedb37ffddfdf8b,17322d3602440d06908cec6f82a93e11,29816dbb517ad29b9a82e28f9629f095]
    POC   21 TId: 0 ( I-SLICE, nQP 16 QP 16 )     894952 bits [Y 46.6647 dB    U 48.6875 dB    V 50.0422 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f87d653b991468c4e87f785330ad7c36,79d41167e50b391c28b99521ca366fc4,8455d497d44bd0fa5a8911c0fb03a92c]
    POC   22 TId: 0 ( I-SLICE, nQP 16 QP 16 )     539712 bits [Y 48.6051 dB    U 49.2100 dB    V 51.0595 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b64ed658a1d178cc62b79f162941396c,7cdc3ccd1d06c9c95afed1a0a542f1e1,eb05af7262122a8dcd2bde473053cfc0]
    POC   23 TId: 0 ( I-SLICE, nQP 16 QP 16 )    1257064 bits [Y 46.8509 dB    U 48.2729 dB    V 48.5642 dB] [ET     5 ] [L0 ] [L1 ] [MD5:bb27853fb2013c82f598dd43dba889e0,f5c9ed17b02a4ef181bcbc88acd823d5,85e143ddea3c774079b17d2475171abe]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   51220.1600   47.0527   49.4584   49.6784   47.7004  

# 15
    POC    0 TId: 0 ( I-SLICE, nQP 15 QP 15 )     965944 bits [Y 48.0473 dB    U 50.6764 dB    V 49.7349 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9f466c724d0ed0f8de1581c4a8e44ea8,cebec24c047993a44ade5ce37917297d,cd6ea5ff873e11b55492bffaed43a128]
    POC    1 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1270704 bits [Y 47.9410 dB    U 48.8333 dB    V 49.1781 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f2bff18aaa28f7ac00c5dbd2e4836532,096caa8665ee07d23912ae0c36aa80bd,33b5c834106b3144233b5b81c2eebb00]
    POC    2 TId: 0 ( I-SLICE, nQP 15 QP 15 )     826656 bits [Y 47.8889 dB    U 51.0818 dB    V 49.4852 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1f2de874b0541924c93b3eb10ba31638,935c194c5e2ea899c434a40f6c01ea2e,e1dafbb7d51d5d1caf376d8adfa835ee]
    POC    3 TId: 0 ( I-SLICE, nQP 15 QP 15 )     725144 bits [Y 47.6258 dB    U 50.4785 dB    V 50.7539 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f628c8423c6d08107ca8158d638143cc,17dada649bb28ff40a177b5297d0d274,8444087e8fc0b0a26a1fba46f1637f86]
    POC    4 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1037616 bits [Y 48.1965 dB    U 49.9710 dB    V 50.5924 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5fb1ad1f4df7d5a2d9b995de81cb0903,e738bbece101d7188df7e7f3559cbca0,aecd656d7c1d3d562b4301af4b4d91ae]
    POC    5 TId: 0 ( I-SLICE, nQP 15 QP 15 )     656920 bits [Y 47.8599 dB    U 51.4994 dB    V 51.3141 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e5e88f1746e60f1f4e40fa65579e70c8,848517330846cbfee66ad2580f79b06f,979c1deec108b863538e40b63e2f13df]
    POC    6 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1679112 bits [Y 48.4093 dB    U 48.2395 dB    V 49.2411 dB] [ET     5 ] [L0 ] [L1 ] [MD5:eedb03021abd15a0717bc488f6bc4ef0,a41c9d7bf8a42678f825ef4a9fce9204,5ea99a2c98333fa884241dd554d22ca8]
    POC    7 TId: 0 ( I-SLICE, nQP 15 QP 15 )     571544 bits [Y 47.9811 dB    U 50.3766 dB    V 51.1125 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4f314739a2fdb959e4036ed4bf0146c6,5315bbd685eb0f6473cb8c25e5f11596,77a184790ae7f0892ce32f77bcae3a53]
    POC    8 TId: 0 ( I-SLICE, nQP 15 QP 15 )     498864 bits [Y 48.2098 dB    U 51.2946 dB    V 51.7900 dB] [ET     4 ] [L0 ] [L1 ] [MD5:2da36faf22e91e9da007ccde09df7e76,13056e8c88c171f2a1dfa502be688f65,fa624f5132a7c71737b7487b9b8105d6]
    POC    9 TId: 0 ( I-SLICE, nQP 15 QP 15 )     765480 bits [Y 47.6674 dB    U 50.5059 dB    V 50.9766 dB] [ET     4 ] [L0 ] [L1 ] [MD5:75a0bedb8aaf060b6db2ea98a6c57726,8469a5ab548c753645f03bc4eba6a5f7,0d326f0659d7e233c9e5363cf0310aa2]
    POC   10 TId: 0 ( I-SLICE, nQP 15 QP 15 )     788568 bits [Y 48.0990 dB    U 50.6663 dB    V 51.3040 dB] [ET     4 ] [L0 ] [L1 ] [MD5:64894816fee25045edaf4b2164f18469,0d508148f8345308c1eac1d40b5fe5fa,4cd65d3f70d4b8cfd519c4148ca3d477]
    POC   11 TId: 0 ( I-SLICE, nQP 15 QP 15 )     826128 bits [Y 47.9996 dB    U 50.4877 dB    V 49.0117 dB] [ET     4 ] [L0 ] [L1 ] [MD5:fe7846110501604a9de52f2220613789,4144a0f9fe0a4b8a22ca06f14b5ae048,3d6899a26f553977f2fb4d750ac9154b]
    POC   12 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1213560 bits [Y 48.0304 dB    U 49.7000 dB    V 49.9656 dB] [ET     5 ] [L0 ] [L1 ] [MD5:24facbdcb8e4a6725d4008a172c7e110,6c086560f4619d1c8a0165267378c861,ad2240a860c8e840e84c0236e70a8d30]
    POC   13 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1302120 bits [Y 48.1303 dB    U 50.2886 dB    V 50.0256 dB] [ET     5 ] [L0 ] [L1 ] [MD5:84c72baad39a18d9cf3383c73e8645d1,766f7bad85851d924689594b0c8c04e4,e5b3638a7cb6bdb5d80ef31f3a8ba946]
    POC   14 TId: 0 ( I-SLICE, nQP 15 QP 15 )     718608 bits [Y 48.1422 dB    U 50.7999 dB    V 50.3152 dB] [ET     4 ] [L0 ] [L1 ] [MD5:72ebdb250324308df2beaa2c492036a5,9485570ea98070f9b5ecea5c10674f67,df30fa98aead9da292d9a6921f0a4472]
    POC   15 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1086296 bits [Y 48.2320 dB    U 49.6477 dB    V 50.0035 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ffcf7f11e33149cdb1a68395b12b20bb,98c09f3ac00b4b9aeffabd2e49276ee6,a7a5e59ec87fb2e967c38c6c8722d91f]
    POC   16 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1283272 bits [Y 47.8667 dB    U 48.6500 dB    V 48.9735 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6f6afc6502e8132229bd114a2ba20f3a,ce291f1507d6cbc174275cc87ddeab69,50509ea35c57fea18082761c02ce9012]
    POC   17 TId: 0 ( I-SLICE, nQP 15 QP 15 )     930584 bits [Y 47.5410 dB    U 49.6740 dB    V 50.1666 dB] [ET     4 ] [L0 ] [L1 ] [MD5:00c37582e240c31003fd6f5673e3e58e,88aa3abb37069253d90458e6615c5ba2,3904871f8e4cadf8d0acab6d66a3796e]
    POC   18 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1035768 bits [Y 47.7377 dB    U 48.8360 dB    V 49.3658 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7bc9b6790899229c518a429956fb67fd,133c2463bcfbf684f6bf8ffd2d97117b,4bdcf7893c05165990d6065cbf8576f9]
    POC   19 TId: 0 ( I-SLICE, nQP 15 QP 15 )     522528 bits [Y 47.6158 dB    U 50.4009 dB    V 50.5716 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4aa3e7a87747d207762b56d350ec9421,d25ea78f1faa4523bed73a78164832bd,ffd238ec0abace519a079bff274343e7]
    POC   20 TId: 0 ( I-SLICE, nQP 15 QP 15 )     699088 bits [Y 47.6795 dB    U 50.8117 dB    V 51.4053 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e1245e53a38a5ca4dbe55f6f64789d2b,726f288417254b7df9751422df7867a1,b3a2ca1dc744e1d5f0f7dc72fd135036]
    POC   21 TId: 0 ( I-SLICE, nQP 15 QP 15 )     986792 bits [Y 47.7757 dB    U 49.3031 dB    V 50.5062 dB] [ET     5 ] [L0 ] [L1 ] [MD5:fadfc45104511907fd8ecb37530ccf10,80894a894fb1ed92ee26b60924158c92,b7cccb62989a98305b820631e7fe04da]
    POC   22 TId: 0 ( I-SLICE, nQP 15 QP 15 )     590136 bits [Y 49.5155 dB    U 49.6663 dB    V 51.5059 dB] [ET     4 ] [L0 ] [L1 ] [MD5:e755ad787911745d88bce16bbbea28ef,408a69e936160c4acee6ed511a64bb63,684166d11dd8cb784afcf6e82963689d]
    POC   23 TId: 0 ( I-SLICE, nQP 15 QP 15 )    1343824 bits [Y 47.8751 dB    U 48.9885 dB    V 49.2405 dB] [ET     5 ] [L0 ] [L1 ] [MD5:2ab0b975a31f96f39a52e717ed16ba3d,ec0bceebee9462581f45ecf47e9fca81,47137a6eb2b10e5e4b90b7c8b3306471]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   55813.1400   48.0028   50.0366   50.2725   48.5761  

# 14
    POC    0 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1043456 bits [Y 48.9504 dB    U 51.1145 dB    V 50.3838 dB] [ET     5 ] [L0 ] [L1 ] [MD5:dcc3644568f85e873b475a9d54568cdd,ccd3bece0c2770c6bd7f6007c863a9f6,3cf0fee641068f7c696b1fce70ed29ea]
    POC    1 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1355056 bits [Y 48.8502 dB    U 49.4529 dB    V 49.7415 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ba20029ad6714a0c1688138b8d2d31f3,4593f027d4ec17859a0d1667d1abbb70,f017ebcc86ad9429b1f96ff62cdf40eb]
    POC    2 TId: 0 ( I-SLICE, nQP 14 QP 14 )     908720 bits [Y 48.8436 dB    U 51.4093 dB    V 50.1416 dB] [ET     5 ] [L0 ] [L1 ] [MD5:70ea67f39099b81ac6ee62bb25f3a095,76754b9bb87d6cab9a7d51e97289db2f,28504f91a89ad6f63293848364362741]
    POC    3 TId: 0 ( I-SLICE, nQP 14 QP 14 )     821992 bits [Y 48.8182 dB    U 50.8504 dB    V 51.0849 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7526df9123ff5940a2853aa7c2fb418f,1c62ac47e813b27f0995c29b2192238a,c2048fa94b8190cc74cff00ac98592fe]
    POC    4 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1121368 bits [Y 49.2252 dB    U 50.4859 dB    V 51.0170 dB] [ET     5 ] [L0 ] [L1 ] [MD5:bde0adfc5155fa98a7d73c328352c9ad,159377e3e22710b03fb45bf906f45cd6,ab36496462813b1452ac4f514fb3fd26]
    POC    5 TId: 0 ( I-SLICE, nQP 14 QP 14 )     736024 bits [Y 48.8129 dB    U 51.8388 dB    V 51.7162 dB] [ET     5 ] [L0 ] [L1 ] [MD5:5b91a1bcb3925aa41093f473c2c36867,6a34eabe8c361b52b6facfb2cbed8bc5,1edab8bd0a5e97f9354faeee2f2f33cc]
    POC    6 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1767408 bits [Y 49.3916 dB    U 48.9701 dB    V 49.7512 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d81579f7929153e7c20a6d52ba4f81c6,431b603b6946052d3871d12a028b1ba9,d846fc0acc842f3268a3c917136b33b9]
    POC    7 TId: 0 ( I-SLICE, nQP 14 QP 14 )     634120 bits [Y 48.6600 dB    U 50.8846 dB    V 51.5611 dB] [ET     4 ] [L0 ] [L1 ] [MD5:28d22ecb9e9329e8b831dd9867d2244c,495527b8e3daf916c5dd52644ad530c9,15167c6bd37587c6d85a8599c0e3d8ec]
    POC    8 TId: 0 ( I-SLICE, nQP 14 QP 14 )     550760 bits [Y 48.8528 dB    U 51.6860 dB    V 52.1843 dB] [ET     4 ] [L0 ] [L1 ] [MD5:55730e4adde9649434ca7bf88ce2e403,568f631848b267bae55c8627d8d880a0,d2e1bbc6b5814740c00e572eb2b723c6]
    POC    9 TId: 0 ( I-SLICE, nQP 14 QP 14 )     859152 bits [Y 48.7588 dB    U 50.9273 dB    V 51.3480 dB] [ET     4 ] [L0 ] [L1 ] [MD5:0dfbd3499b6b2f98a5fda04919ffbe6b,652e941f03adfbf6de90be5e7ba46854,ea03c28d6a2d0513ca2db2fd32572799]
    POC   10 TId: 0 ( I-SLICE, nQP 14 QP 14 )     855760 bits [Y 48.9344 dB    U 51.0627 dB    V 51.5878 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f39303c0fa37c75a96185c687e1edf59,1d75275685811e841dcb3605d326ceb8,6547f31f7896d03515d854b863a01baa]
    POC   11 TId: 0 ( I-SLICE, nQP 14 QP 14 )     910296 bits [Y 49.0016 dB    U 50.9100 dB    V 49.5900 dB] [ET     5 ] [L0 ] [L1 ] [MD5:753155fb9fe50f8ea287403d9ca6c516,ee694294f427d198d4b986c6fce546e8,3156be338d63fc2ab2d910852371b002]
    POC   12 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1298576 bits [Y 49.0343 dB    U 50.2427 dB    V 50.4594 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8beee3de8a79e09ffc81f6b4bbb90b43,63a446283334e780687421a234d1fa77,b97606c17e6e1caf294fe4b759239e01]
    POC   13 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1382560 bits [Y 49.1043 dB    U 50.7124 dB    V 50.4502 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1f7804003e9e13f410832e88fd9f495d,42b84be0850151118b438b28a77e89a8,39b9014302a4ed6a8a4e79bed44f40c4]
    POC   14 TId: 0 ( I-SLICE, nQP 14 QP 14 )     796240 bits [Y 49.0883 dB    U 51.2533 dB    V 50.8542 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4c02e4eb0f0f21753d2eb8dc575e6fe1,67118ea34c36c3a27c9c78388b413b5e,89308cd4df06f4d987b00d0986796b1e]
    POC   15 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1165488 bits [Y 49.1603 dB    U 50.2606 dB    V 50.5443 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f8c9a3d84fd9242ee283b90474470eb0,a2523629bdeda9f63242977411795383,b18ac61479dbdb5fbce78ea073c76197]
    POC   16 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1382184 bits [Y 48.9998 dB    U 49.2640 dB    V 49.5755 dB] [ET     5 ] [L0 ] [L1 ] [MD5:80f6cbb230b019ea55a47520c0fd6393,1947422672a84a52a96fac71e440fdb2,d953407ae82f14002245ee63b2a2560e]
    POC   17 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1022560 bits [Y 48.5770 dB    U 50.0961 dB    V 50.5768 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6dd93e0cc9b1d2bfb169b6862c08985b,76dac7c5b60e6bf8a3159991c8384c3f,b04c4a09767bd55452f1d43b7fadbc92]
    POC   18 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1134464 bits [Y 48.8160 dB    U 49.4294 dB    V 49.9618 dB] [ET     5 ] [L0 ] [L1 ] [MD5:63bfa23be78e733da68c598c5e1b1cb1,c37cee37c2e13752a9e29199330c3e61,43c2adbf642f7d8d9d5675a43d4c1b64]
    POC   19 TId: 0 ( I-SLICE, nQP 14 QP 14 )     610088 bits [Y 48.5563 dB    U 50.8643 dB    V 51.0596 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5d1aa0e99a672a060895824bce9dc797,7103efaf98db3f350afb3af6461fa722,c943dc474407ae1f7beec67221234240]
    POC   20 TId: 0 ( I-SLICE, nQP 14 QP 14 )     796256 bits [Y 48.9190 dB    U 51.1877 dB    V 51.7326 dB] [ET     4 ] [L0 ] [L1 ] [MD5:49915901604a4650276e351acd8846fb,95abbbaff887228b173925ba696c4c26,416698b64d0f5bd78e69f49440187a3c]
    POC   21 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1086296 bits [Y 48.9551 dB    U 49.8187 dB    V 50.9188 dB] [ET     5 ] [L0 ] [L1 ] [MD5:524bae6981f4e5bc1ccb499b4a2a16be,68b96cf94788739bf8f377ace1c3c56b,9338685e4f7f5f5ee83a2cd6e5bd3304]
    POC   22 TId: 0 ( I-SLICE, nQP 14 QP 14 )     654176 bits [Y 50.5488 dB    U 50.0426 dB    V 51.9700 dB] [ET     3 ] [L0 ] [L1 ] [MD5:a22c2f3e778c857cadf3f77f7fc9ff9a,9f9ce9e5301b60e59c310c540095ad6e,07a8e9e05b7d76ddcb97756e547ebf11]
    POC   23 TId: 0 ( I-SLICE, nQP 14 QP 14 )    1436248 bits [Y 48.9331 dB    U 49.5427 dB    V 49.8416 dB] [ET     5 ] [L0 ] [L1 ] [MD5:cc53bfc02d4ba9acedbf99b42899721c,5485f11b6bac6d2aeb5c9751815cdcfe,580e6bc3727434867f027d7f29beb64f]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   60823.1200   48.9913   50.5128   50.7522   49.4432  

# 13
    POC    0 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1130976 bits [Y 49.9386 dB    U 51.6409 dB    V 51.0149 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7714ee7c42602b1be1b09475029ce446,49b384ae7656888e305f3e1a295f9807,6e6e95ae3d92c5dcd8423d4bfb15b098]
    POC    1 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1448992 bits [Y 49.8456 dB    U 50.1018 dB    V 50.3762 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c3cd339cd044da94c53c3e88212f700f,ab745bbaf307a9645b8f681b98820cdc,3a28df4b1a8c773036bd1dd05e6666aa]
    POC    2 TId: 0 ( I-SLICE, nQP 13 QP 13 )     993408 bits [Y 49.8001 dB    U 51.7222 dB    V 50.7726 dB] [ET     5 ] [L0 ] [L1 ] [MD5:684e68c39018cec19363048392b9232f,ba2c916e89763d2249f23af7479ee1de,ae8bade97047a5ed4e40439cb7fdfda2]
    POC    3 TId: 0 ( I-SLICE, nQP 13 QP 13 )     911832 bits [Y 49.8891 dB    U 51.2925 dB    V 51.4729 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ccf2a967fe19e89095f3a61e2f1f13fd,1dae963674708f8da63e6ab548ed9866,668d479626a6f1eeded976770055f8fb]
    POC    4 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1207344 bits [Y 50.2456 dB    U 50.9834 dB    V 51.5237 dB] [ET     4 ] [L0 ] [L1 ] [MD5:25605e8b169a26a56ef2324f60394a55,1c530b249435e9cb0cf64dcc83e747c2,93f46701862b61be6b71140938b472e7]
    POC    5 TId: 0 ( I-SLICE, nQP 13 QP 13 )     819128 bits [Y 49.8031 dB    U 52.2007 dB    V 52.0761 dB] [ET     4 ] [L0 ] [L1 ] [MD5:3f427fa8554cb9fcb8ca003961a7409f,f58863494d4ddf1f8a074e79a16ba1cb,8481a557001fac54e60a2343c6bf4da4]
    POC    6 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1866528 bits [Y 50.4242 dB    U 49.7110 dB    V 50.3130 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d53289c0257ec2c37d80b948f405d711,0ec8b0e8f1ee493e1f920ecaa4d5125b,1820f75f5a0ef0260e74af8edf409cfe]
    POC    7 TId: 0 ( I-SLICE, nQP 13 QP 13 )     715592 bits [Y 49.5010 dB    U 51.4366 dB    V 52.0621 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f5c10d12193b8468c426c7332b9721e5,76a31e8012d040dcb115cb999e59d140,57ba615c61a7c71f53b24da168306098]
    POC    8 TId: 0 ( I-SLICE, nQP 13 QP 13 )     618624 bits [Y 49.6139 dB    U 52.0730 dB    V 52.5487 dB] [ET     4 ] [L0 ] [L1 ] [MD5:5e534cff5ff50ff9d61e0585a7fd588d,e9522b40d857134c49c7b75ffa50caba,b3fea1d96a80ab7725c1e1e157f05851]
    POC    9 TId: 0 ( I-SLICE, nQP 13 QP 13 )     951048 bits [Y 49.8275 dB    U 51.3657 dB    V 51.7408 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7f57f11709aa5ee41eb818f9b1b1ab79,69be3487f2cdb910e0287aa6513fd59c,f4b97e1730119882ea92c2ad85895472]
    POC   10 TId: 0 ( I-SLICE, nQP 13 QP 13 )     936720 bits [Y 49.8561 dB    U 51.5168 dB    V 51.8721 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ad5cb287e24c718521e376366e145571,3ac3cdc620914d16daa530c176e09f55,748f78a2a1bc01486633771db15fb741]
    POC   11 TId: 0 ( I-SLICE, nQP 13 QP 13 )     997992 bits [Y 49.9915 dB    U 51.3764 dB    V 50.2389 dB] [ET     5 ] [L0 ] [L1 ] [MD5:64868c750a5908c645a663868ea8c148,d1929d5a416b3da0f6dd610a63b2b515,291dd0c87f88e30725540e0132f330ad]
    POC   12 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1389240 bits [Y 50.0616 dB    U 50.8028 dB    V 50.9779 dB] [ET     5 ] [L0 ] [L1 ] [MD5:abfcda10660e831acafca2b33e4d9dbe,f42e4a02db2a93b39f4be7aa6dd0a776,f6cdff13fe13b7921407524b30bc7a40]
    POC   13 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1470624 bits [Y 50.1129 dB    U 51.1875 dB    V 50.9681 dB] [ET     5 ] [L0 ] [L1 ] [MD5:86cee4fd092348b2edfa8b39cb217501,250c786b309685ec833ca366e6727314,db14f0e8a212298cc41955bb82501bfd]
    POC   14 TId: 0 ( I-SLICE, nQP 13 QP 13 )     876344 bits [Y 50.0178 dB    U 51.6779 dB    V 51.4290 dB] [ET     4 ] [L0 ] [L1 ] [MD5:12bb226fd3cc0c6d04d2a1d2b98057c9,33b5e7d3a709858bae9de636757782c5,46551619e388bf3fdea3e60d97e570e9]
    POC   15 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1250480 bits [Y 50.0905 dB    U 51.0024 dB    V 51.0999 dB] [ET     4 ] [L0 ] [L1 ] [MD5:96179aded7a6c6fc6d423bd1ac43f3dd,0aac6814716c1f259635cc7e99a1966f,f34fdfdbfcf429c9e830dda777f4ab73]
    POC   16 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1478256 bits [Y 50.0251 dB    U 49.9371 dB    V 50.2133 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a4fc407dc5fe15f88f8dbc452cd985ce,462797aefd3496c2cf950836a9d09a91,62e171ae5489b6d55735b57277ac9246]
    POC   17 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1117544 bits [Y 49.6694 dB    U 50.5854 dB    V 51.0910 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d05d70438589ee8c9c06f6df1d2e11c3,55351f292fdf4fb18ebeb9d078de4c85,ed8df7643e054d51f59e90d4095b3374]
    POC   18 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1233256 bits [Y 49.8619 dB    U 50.0618 dB    V 50.5910 dB] [ET     5 ] [L0 ] [L1 ] [MD5:963fe37beca3a6cc666eaeb15cf154c6,e5d6ed632d30aeb45ebfd33a02e79f51,a986164f641c501fa5859d8ec25bf332]
    POC   19 TId: 0 ( I-SLICE, nQP 13 QP 13 )     695952 bits [Y 49.5091 dB    U 51.3484 dB    V 51.4985 dB] [ET     4 ] [L0 ] [L1 ] [MD5:d421d04c741fd4d1418390b1e6910f2c,f44e60a97af3d2820f5a25836620d84d,1636d9e080351d5c5a142f2446aa81d8]
    POC   20 TId: 0 ( I-SLICE, nQP 13 QP 13 )     880288 bits [Y 49.9624 dB    U 51.5577 dB    V 51.9810 dB] [ET     4 ] [L0 ] [L1 ] [MD5:1eeaf51bf052ff2e896f096d57e22db4,10a0a91a10cdd29b4fd59eb75270007e,3b767ff761863fceb26c8cb87815e00c]
    POC   21 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1176008 bits [Y 49.9999 dB    U 50.3572 dB    V 51.2754 dB] [ET     5 ] [L0 ] [L1 ] [MD5:904fd60866af4de54d41af5260b00fce,708fc15faf24499312055af1eab52a1b,26d897cbea1b8d3d5d226663f8ca0f00]
    POC   22 TId: 0 ( I-SLICE, nQP 13 QP 13 )     706752 bits [Y 51.3845 dB    U 50.4939 dB    V 52.3527 dB] [ET     4 ] [L0 ] [L1 ] [MD5:a53ff3982a13939c50ad7ed6ea69be9f,45a15ca7ead77ee3642ff6b646151eda,564c39d476df959ea6f4f69305fa8f69]
    POC   23 TId: 0 ( I-SLICE, nQP 13 QP 13 )    1530376 bits [Y 49.9883 dB    U 50.1315 dB    V 50.3916 dB] [ET     5 ] [L0 ] [L1 ] [MD5:822692f9d25902b44c7a6885bfd93b51,8eec33d2b143f229260f3f4e70a46cee,26c57003bde8a0dc6800453deb2d423c]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   66008.2600   49.9758   51.0235   51.2451   50.3032  

# 12
    POC    0 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1223536 bits [Y 50.9320 dB    U 52.1688 dB    V 51.5853 dB] [ET     5 ] [L0 ] [L1 ] [MD5:08d46256ff3dfa6ccc99830825457132,12fc44199945b87120ed88640e5a58b6,8a99bea95cfe4e45685b468104030f53]
    POC    1 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1545952 bits [Y 50.8008 dB    U 50.8119 dB    V 51.0407 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ab6939ba5f60b800e67d708b40567009,c5d6a441c7fb9890b7a8c56bee64f29b,ab0499da64c1de98486f2c50d78c48c3]
    POC    2 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1087896 bits [Y 50.8066 dB    U 52.0982 dB    V 51.4733 dB] [ET     5 ] [L0 ] [L1 ] [MD5:17b2f5a53df3326d1c3d48f83207763b,47a57487555523dd4b56b225ff5e6c1d,751c994e42674f0f9f55e318bcee1ab0]
    POC    3 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1003248 bits [Y 50.9172 dB    U 51.7420 dB    V 51.8794 dB] [ET     5 ] [L0 ] [L1 ] [MD5:69e327c89d30051c26bcdc6d38ec372d,1b7154fc38417b9e825df38943ff2151,81892eaec87384dea748873a85bad1ee]
    POC    4 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1293248 bits [Y 51.2130 dB    U 51.5634 dB    V 52.0857 dB] [ET     5 ] [L0 ] [L1 ] [MD5:dac35817ba99324efe341b01a8a0d4b3,dfd68f642a3e72ac68967041bfaf53b0,8b7b5ff1d022493d5cbe61dda52dc072]
    POC    5 TId: 0 ( I-SLICE, nQP 12 QP 12 )     910888 bits [Y 50.8575 dB    U 52.5689 dB    V 52.5010 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8d58a67e0cb3e4ab61ce4c552195b72a,fe59d0c22056ba3894835ea3dcbb0dab,bac32eb73c9c56cea3ba6cbb16b34cd2]
    POC    6 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1964192 bits [Y 51.3857 dB    U 50.5030 dB    V 50.9567 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1ce68e79fa22c5b4e6e860edc7d669bf,01574d0347e076e13d2e75e398e7aafb,33c311c1b69f197fc643a8452ce4266b]
    POC    7 TId: 0 ( I-SLICE, nQP 12 QP 12 )     811680 bits [Y 50.4767 dB    U 51.9336 dB    V 52.5464 dB] [ET     4 ] [L0 ] [L1 ] [MD5:33a5568e0fb0d144938ddc3335c3fd4e,8b2165a00987e36aa7c755a786a162c9,30223a0b0ede6269cf7e2072033e2ae0]
    POC    8 TId: 0 ( I-SLICE, nQP 12 QP 12 )     700136 bits [Y 50.4858 dB    U 52.5131 dB    V 52.9309 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a557a5247147de9bc91530faaf27d73f,90b8f7c9f4bdda04d2b47f6a0f2d8405,a1ece40086a2a7d7d33cfa1b9b6772a2]
    POC    9 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1045864 bits [Y 50.8940 dB    U 51.8538 dB    V 52.1516 dB] [ET     4 ] [L0 ] [L1 ] [MD5:7f7c2e31729459e6078560806d199f0c,afda393a89abf8a9c03ae133a17e16c3,7c3a82f422d4697d046fc3d70c583a64]
    POC   10 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1027472 bits [Y 50.8569 dB    U 52.0202 dB    V 52.2772 dB] [ET     4 ] [L0 ] [L1 ] [MD5:da1b5537e09138a2313bc4ca9cf9c0d8,7049afcb1d6c1a47b9b990e0618a57e4,d9bfeb61f2ad4028f4528ecec29927ea]
    POC   11 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1091552 bits [Y 50.9782 dB    U 51.8914 dB    V 50.8616 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b8f1c0cc2be78ddf641020afa132c0b4,2c4a9e22be7a168201ba446a2e2be7b9,c4d71b30784b114211ce973df1a3adc1]
    POC   12 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1481624 bits [Y 51.0505 dB    U 51.4272 dB    V 51.5557 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8e2ae99e7418e080be30020c24cdafa6,563fb6b4cacc3b4f7534fd0034d69cd1,8a076db6b1770314455c8448e95273ad]
    POC   13 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1560504 bits [Y 51.0804 dB    U 51.6694 dB    V 51.4855 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6874d704eee791a68b34b8f202b5a69f,27ff9f0da59c5f3534a578a39d306115,285a65c2deac581197b200457007cd6e]
    POC   14 TId: 0 ( I-SLICE, nQP 12 QP 12 )     957560 bits [Y 50.9319 dB    U 52.1390 dB    V 52.0114 dB] [ET     4 ] [L0 ] [L1 ] [MD5:eb89e03074a7e3d8057c1fec0d975022,5eeb8d336fb5be038b032d06d8c754ca,0a09f7ca5ccf02540642134c7b0613c1]
    POC   15 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1339376 bits [Y 51.0167 dB    U 51.6247 dB    V 51.6805 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a1b858d592e59f152f3e2181f1f72a6c,dea0277b20ef7aa7c8a81417edd87041,f302a1feddbb1971892417071416f616]
    POC   16 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1578824 bits [Y 51.0373 dB    U 50.6449 dB    V 50.8659 dB] [ET     5 ] [L0 ] [L1 ] [MD5:57ad70d77d09b6ff97b8b94dcf1f0657,ac6ebb94698db3830ed2a871bd6b4943,80c15d21bf6eb3ef97837c8037ecb301]
    POC   17 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1221416 bits [Y 50.8376 dB    U 51.1149 dB    V 51.5872 dB] [ET     5 ] [L0 ] [L1 ] [MD5:fe75e12991a667bff2bdd98fed166b7d,8794bb0caee79ffff6b0cb53c91bafe5,ce21a3f199f960cc3cad1e8706045093]
    POC   18 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1334008 bits [Y 50.9113 dB    U 50.7066 dB    V 51.2039 dB] [ET     5 ] [L0 ] [L1 ] [MD5:08f52e385e306cab934d50416f479c0f,9906736f0c49194d905bab05157198b8,3cb5a227b277ce0ede922d8ec4ef97ba]
    POC   19 TId: 0 ( I-SLICE, nQP 12 QP 12 )     797448 bits [Y 50.5932 dB    U 51.7875 dB    V 51.9639 dB] [ET     4 ] [L0 ] [L1 ] [MD5:84bc70b4dbc30f8f0dfb2f20b664c7f9,acbcac458f80be83b434df524f2e24ed,422c4a7fceec706185fbfe9005d120d4]
    POC   20 TId: 0 ( I-SLICE, nQP 12 QP 12 )     969328 bits [Y 50.9963 dB    U 51.9673 dB    V 52.3599 dB] [ET     4 ] [L0 ] [L1 ] [MD5:42c04366dc06c4ee0a14948f10444a19,dbe18728d1fbe0bbfa4f10e50c6d989f,1ebea2cd81e4fad79a11392cea50fea0]
    POC   21 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1268840 bits [Y 51.0181 dB    U 50.9209 dB    V 51.7467 dB] [ET     4 ] [L0 ] [L1 ] [MD5:308659556de684f21eef087a2e143b42,0e8863d69469ffe201beb4614ed85d60,ac98445765b968c5ecd73adad4f32166]
    POC   22 TId: 0 ( I-SLICE, nQP 12 QP 12 )     773360 bits [Y 52.2533 dB    U 51.1292 dB    V 52.8014 dB] [ET     3 ] [L0 ] [L1 ] [MD5:3f4d1f7441f1bde7f8312826eb9102d7,e46f3fb684c36ef793a170ef057b7408,b4c0faabc25c04dbcf1def18d474a56f]
    POC   23 TId: 0 ( I-SLICE, nQP 12 QP 12 )    1624240 bits [Y 50.9561 dB    U 50.7748 dB    V 51.0141 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6bc9dd02e40c4725614c09e1dd3e5dfd,482cd5d5f843c65c160cb667319eee72,11e54a0cb5d3fdc208fd89da351177f2]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   71530.4800   50.9703   51.5656   51.7736   51.1702  

# 11
    POC    0 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1316664 bits [Y 51.9352 dB    U 52.6565 dB    V 52.1557 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6af0c783807554e8556a27226d351e13,6d91ab827936da94078ca2086007e48c,54ba32bc07cb91f452c6c490cf971ecb]
    POC    1 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1641408 bits [Y 51.7009 dB    U 51.4906 dB    V 51.6706 dB] [ET     5 ] [L0 ] [L1 ] [MD5:072960036296e3d174127a952256e253,112c829f51ba05a6f98a19c54b4798d4,5cb414bff751ee3ab54ded9214cf1551]
    POC    2 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1180072 bits [Y 51.7738 dB    U 52.4713 dB    V 52.0823 dB] [ET     5 ] [L0 ] [L1 ] [MD5:fd21c03bb359fea5c8ab0b34a25e66c7,fbe77deb26c4283536eca62109eb4fde,fedf199b2a9d2304e172bbdbb8101ebe]
    POC    3 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1091080 bits [Y 51.9388 dB    U 52.1527 dB    V 52.3189 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c67da73e31eeccd70eeabf02d4c5c898,90a2815c751ae689516f8b04e0928571,8dd43aecea547ed6f81fbbddfaf47278]
    POC    4 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1379200 bits [Y 52.1982 dB    U 52.0319 dB    V 52.5595 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d810c61b994bcaf0abf1c5b1b7612d5d,56a7d88ed0878bde0df1d7342310cc44,c3dc7004348b165a908d42f4d62ee050]
    POC    5 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1003152 bits [Y 51.9165 dB    U 52.9513 dB    V 52.9120 dB] [ET     5 ] [L0 ] [L1 ] [MD5:722674ee16b52f30e9c6f26678fb20ca,4dc8153236d603d7b51603a9f63084aa,0666037b6de21c846502767fc44de5a2]
    POC    6 TId: 0 ( I-SLICE, nQP 11 QP 11 )    2058584 bits [Y 52.2617 dB    U 51.2621 dB    V 51.5797 dB] [ET     6 ] [L0 ] [L1 ] [MD5:b52686868c2d4aaa9d5ca55c475b35b6,9adb8ef669e9b163ff9a67167011b3cf,cb8b34c52e337cbce4c7493832c62b13]
    POC    7 TId: 0 ( I-SLICE, nQP 11 QP 11 )     914632 bits [Y 51.5458 dB    U 52.4316 dB    V 53.0860 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0a57aac89ec119449a77aef37f1effe8,084a1b64e694202fb590bcc358b4cb5e,c2fa3c2d0da65dcb2f9ddf0d9a729c91]
    POC    8 TId: 0 ( I-SLICE, nQP 11 QP 11 )     802000 bits [Y 51.5518 dB    U 52.8790 dB    V 53.3366 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0fe6513457b14a2a08a5f3d92e651f72,d888e7d483acdb4b7f93a31f5837f710,87928372cd6657c9a1e6d4f402c2d511]
    POC    9 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1136984 bits [Y 51.9527 dB    U 52.3152 dB    V 52.5276 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b061336326cb4bda943b716da37ca691,5930034413afc4b98526e13246cff746,abaa24d70e1cc8c034dac52ac0958f1b]
    POC   10 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1118368 bits [Y 51.8641 dB    U 52.4993 dB    V 52.6398 dB] [ET     5 ] [L0 ] [L1 ] [MD5:2b40b9c86d65a90ab4464e359da40200,44614655c97f6adad9b68f5564bb991b,c4dd364890a8d1bf3c82a4a998cf567a]
    POC   11 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1187520 bits [Y 51.9773 dB    U 52.3812 dB    V 51.4990 dB] [ET     5 ] [L0 ] [L1 ] [MD5:34a772abb351540b45abca66d02207f3,f3b3c9ae7a19c9ace40d2a26abcd0e4c,26af6c578f532aa5ca4e04f5271c345e]
    POC   12 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1571920 bits [Y 51.9808 dB    U 52.0117 dB    V 52.0511 dB] [ET     5 ] [L0 ] [L1 ] [MD5:dd8cb3a9801949755c92e5aaa436601d,ba5f6f6b319dc640f0731968612c4390,68337bb2b1f567e8002d3b83af906ce6]
    POC   13 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1643496 bits [Y 51.9874 dB    U 52.1414 dB    V 51.9850 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c3fc6126471df6aa17024274165e7f46,c6fbbc714e86d59340e1daaf16dcfbb4,c59f7ae2cb867a69f916a1e9fd6ce614]
    POC   14 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1046776 bits [Y 51.9063 dB    U 52.6160 dB    V 52.5102 dB] [ET     5 ] [L0 ] [L1 ] [MD5:306902203d38dce2ba57859585bc7440,ff99985356afddaa3a8ba30dda2d356d,86a12b5732e139d24a202d9dd02204df]
    POC   15 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1428552 bits [Y 51.9558 dB    U 52.2778 dB    V 52.2465 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b42293887bad954a539e779451bddd0d,26f11e7a85a5f77eb13dd45055e8f73b,00d0067631e81023e51a8902ff5a521d]
    POC   16 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1675944 bits [Y 51.9876 dB    U 51.3404 dB    V 51.5068 dB] [ET     5 ] [L0 ] [L1 ] [MD5:83ed9735b260162806c35e12e5593fd7,d25abd0d3f801b971cdc3a1c9300a14b,bb1300c513fb23f5b0b31a65d12d35a9]
    POC   17 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1316304 bits [Y 51.8482 dB    U 51.6448 dB    V 52.0739 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f7a0fa30abffd812de19e38f4c649c69,4101d03c5510545ee5d94d3197e427a7,aa98473056d5e880bf74460e97cf9388]
    POC   18 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1431520 bits [Y 51.8861 dB    U 51.3752 dB    V 51.7869 dB] [ET     6 ] [L0 ] [L1 ] [MD5:97a36f25ff9818897ce634c76179a743,9235243fe4727f8ad83d5265e4e625a1,7ed4410c2ba28391b6af6771785a7f9e]
    POC   19 TId: 0 ( I-SLICE, nQP 11 QP 11 )     896264 bits [Y 51.6748 dB    U 52.2602 dB    V 52.4407 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e2c7b84492298a3f48ec0456eb980064,74ab993ed9209d50175a20dd68629845,2216c2023c95aa91215ec7d54f3c67cc]
    POC   20 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1057904 bits [Y 52.0401 dB    U 52.3627 dB    V 52.6969 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f04af86d6d87778ad9e93d5f9463bb52,b97af75d8cfa863c62f617cc05a3774d,cbc9e11795aeee74cd47cdbdd64b2311]
    POC   21 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1363560 bits [Y 52.0722 dB    U 51.5039 dB    V 52.1347 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5ae92570b4837f6a54ade67fd600b962,d2d381a02a6e92ece89b23d8c1599d7f,5442dbb296a5d910c143875df3e97034]
    POC   22 TId: 0 ( I-SLICE, nQP 11 QP 11 )     844840 bits [Y 53.1629 dB    U 51.8470 dB    V 53.2841 dB] [ET     4 ] [L0 ] [L1 ] [MD5:3d147d250edd355658c58f2a8f3de454,8dcc54299530df9cd97d619a5ff10e45,564898f6e99cb3ec51745b6c4e7d4cf1]
    POC   23 TId: 0 ( I-SLICE, nQP 11 QP 11 )    1717576 bits [Y 51.8747 dB    U 51.4062 dB    V 51.6354 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3e8bcfb24246412a1f9be87919fdb7b2,f2034d41507c8d6f01b33f772053ea8f,4f8b5a28e84f84394b268bff10f331fd]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   77060.8000   51.9581   52.0962   52.2800   52.0167  

# 10
    POC    0 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1410392 bits [Y 52.9313 dB    U 53.1600 dB    V 52.7729 dB] [ET     6 ] [L0 ] [L1 ] [MD5:48c39d98b4472b8bd953b193d6bdc9cb,94da16834b1081e512adf421bc1cb5dd,f12ccbde0591095b00afb5be24c9e512]
    POC    1 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1746760 bits [Y 52.7028 dB    U 52.2277 dB    V 52.3481 dB] [ET     6 ] [L0 ] [L1 ] [MD5:6a35230e8c38dcebb842c0e45dd0d61a,ec7fafdcd7024b65855102065eab2658,a6ab4fa82232a6e713a46f4f1abe57ab]
    POC    2 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1279504 bits [Y 52.8065 dB    U 52.8455 dB    V 52.7429 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0f2d9da15053b13d8a11955a28463367,4643663565ace553bfefd0c1feb72b54,3308793a8dd83e462acf4d60f81752bd]
    POC    3 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1181064 bits [Y 52.9174 dB    U 52.6105 dB    V 52.6994 dB] [ET     5 ] [L0 ] [L1 ] [MD5:74b38fb11e5868ea57379e8946616f83,157b1a9bf7f03cfd2c8f0751cc6efb2f,30b2573cfceeef00dbd891564f1a66bf]
    POC    4 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1469048 bits [Y 53.1564 dB    U 52.6660 dB    V 53.1492 dB] [ET     5 ] [L0 ] [L1 ] [MD5:01ccc01eb6cbb0f56d2d35e8c4880a07,0d04efdd1c6cf52764bae976d6574492,b805d4803fdf90a8fc54da773f7a7dd2]
    POC    5 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1090056 bits [Y 52.8809 dB    U 53.3266 dB    V 53.3694 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e8dee0bfd8d664e19b3b22f8e71d4fdc,043a29f493f995abdf516757c3505b50,c5d95fff4d4d614a4e54c1439753810d]
    POC    6 TId: 0 ( I-SLICE, nQP 10 QP 10 )    2163928 bits [Y 53.2495 dB    U 52.0709 dB    V 52.2599 dB] [ET     6 ] [L0 ] [L1 ] [MD5:eb05408d7f5c25ca718401aabdeee6dd,26e9190e9024ef8ffea06e6f0a0f8571,f6958346bd1b456011f0e2fce1082dee]
    POC    7 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1012688 bits [Y 52.5254 dB    U 52.9440 dB    V 53.6128 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d193e6b799fd0058e48de6e5770a3d7a,986d9ca570916fc65fdb912a6fa3722f,4bb3d0bd3b86c1eda5a1d537f53b49de]
    POC    8 TId: 0 ( I-SLICE, nQP 10 QP 10 )     898192 bits [Y 52.5775 dB    U 53.2941 dB    V 53.6928 dB] [ET     4 ] [L0 ] [L1 ] [MD5:9daa24deda0f47bc0babfd009595777e,4bf447cf5a51a95dd8608365c6bbe661,980a85ba0d3628de1f6b643785e45719]
    POC    9 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1227720 bits [Y 52.9373 dB    U 52.8009 dB    V 52.9947 dB] [ET     5 ] [L0 ] [L1 ] [MD5:effd08e89d0150a7221518f6bb74bf8e,2e639fc5488b72cb93ec1f7eaeda1bbf,f6345bda4dc2875bb1e604b02b9fba83]
    POC   10 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1214304 bits [Y 52.8585 dB    U 52.9966 dB    V 52.9601 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f2371f0c7d9322af2b2a713e3a5dd5bb,c6914f99bdccc82a1cc30c1a1569b139,4c9cbe8913a8e9f3ed3329605dd454c0]
    POC   11 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1281192 bits [Y 52.9615 dB    U 52.8669 dB    V 52.2076 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a377e9a656123be63b1bfaf026520bce,67c3eec95b02f75787eb7adb5f86b680,f17ce0b75bb4b398bd33abe2746d9170]
    POC   12 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1669352 bits [Y 53.0017 dB    U 52.6438 dB    V 52.6356 dB] [ET     6 ] [L0 ] [L1 ] [MD5:caa480a2b510af273364f3bd54f2d7ea,ebe398472e33048aed0da632597dca77,edbce3ff6358a6156634108c290913b5]
    POC   13 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1740504 bits [Y 52.9940 dB    U 52.6552 dB    V 52.5205 dB] [ET     6 ] [L0 ] [L1 ] [MD5:fc695304c03d23607d21b6dcdb5423a4,4bb63805ae2e52a444f4553b3e1600c1,4a3ec8a9fc1d1f0598609d2c34b5ffde]
    POC   14 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1137568 bits [Y 52.8365 dB    U 53.1026 dB    V 52.9894 dB] [ET     4 ] [L0 ] [L1 ] [MD5:ed62345676c21d74c3529eda600671d0,b1f4dde3919787caf15cdd4f6200e9d8,2c723471ff4ff5b2da3dd092bcf7a079]
    POC   15 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1524480 bits [Y 52.9275 dB    U 52.9669 dB    V 52.8556 dB] [ET     5 ] [L0 ] [L1 ] [MD5:67c21ce83a9891cc000558c51e0caf52,c9d006122f097bdd2c27930b9488aea5,091bdaf3002dd2fa2e6f3a68f8173e46]
    POC   16 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1782912 bits [Y 53.0272 dB    U 52.0900 dB    V 52.1409 dB] [ET     5 ] [L0 ] [L1 ] [MD5:45f9441ad9306c2e5b632d7d9b95b8c0,47aa4b9b7db2817965fb16c575400906,21159aa3fd3dc19a4c7f88bb66ff685e]
    POC   17 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1419416 bits [Y 52.9224 dB    U 52.2280 dB    V 52.6235 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5ec924e1a3570a8fabf8f1f990c8db9c,fbf3236042d3aed280b9165beba2c345,c0aa3c60aa662cdf2f05129b419fd1f2]
    POC   18 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1538000 bits [Y 52.9133 dB    U 52.0909 dB    V 52.4380 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ad9684fa71d4ae1114e7ec1080a1f795,91e504d58cb11e114629a04a2ee031d2,53673a138e8f76f833438b2437a24ea5]
    POC   19 TId: 0 ( I-SLICE, nQP 10 QP 10 )     995424 bits [Y 52.6883 dB    U 52.7617 dB    V 52.9124 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7785ad490f5f6d5675f5d1becb3e6fc3,449aae1895edf37ba69a4b9bc0a264b9,39e7944f07574b3f0d330bd523a091b8]
    POC   20 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1142832 bits [Y 52.9914 dB    U 52.7757 dB    V 53.1029 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b712fdd3ab7aca64632394394577135c,8135b6e84d2dfd5c93251e45c820c91a,871c9263c972e7121bc6d3cb964dd2c4]
    POC   21 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1460960 bits [Y 53.0332 dB    U 52.1684 dB    V 52.6356 dB] [ET     5 ] [L0 ] [L1 ] [MD5:347ddfa06101eff41dcf35ad0d411128,4e562b3accedbe3b57a03ce15f4aa0e2,7d592a874500cd62379727b805c24c23]
    POC   22 TId: 0 ( I-SLICE, nQP 10 QP 10 )     914968 bits [Y 53.9662 dB    U 52.5973 dB    V 53.7839 dB] [ET     4 ] [L0 ] [L1 ] [MD5:3b2a514e87a269788b8c5e745e3c2f68,3cd111650a9d0099f38ef82fa24b9755,3a0ab3d14ae15f4bce99a848de639367]
    POC   23 TId: 0 ( I-SLICE, nQP 10 QP 10 )    1819536 bits [Y 52.8744 dB    U 52.0929 dB    V 52.2767 dB] [ET     5 ] [L0 ] [L1 ] [MD5:44cd950a2946a1758c8229b35ecd7dc0,bbf93d3344aea5e3484ae0632e7d45dd,53b73cf688b3c52ab88c0cd7e84ff277]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   82802.0000   52.9450   52.6660   52.8219   52.8646  

# 9
    POC    0 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1512856 bits [Y 53.9775 dB    U 53.7310 dB    V 53.4579 dB] [ET     6 ] [L0 ] [L1 ] [MD5:8d1150fd07074e0acb692e7c2215aed1,333020907b79447a30a7291d28aa67fe,48d69b48af257cae7a34a44fc5d0a8c2]
    POC    1 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1861200 bits [Y 53.7581 dB    U 53.0022 dB    V 53.1183 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f4b8405ed30565dbd7a00758eab1cf22,2d9a5e2381da6e1f8eb8dff74147e7e6,ee81b1c7f1947b7f7768e94745263e65]
    POC    2 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1386216 bits [Y 53.8753 dB    U 53.4390 dB    V 53.5099 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8e7062c51b369224bac392ecfabf3c51,24baf4abd8d049fb31c317f21f8fbc38,7d8fd473b313a2c469db39cb699ec198]
    POC    3 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1284464 bits [Y 53.9729 dB    U 53.2154 dB    V 53.2608 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0809dfde763c08ed00fb50d2df05942c,d8c8cfc5ddd5431e2e12c3a148064e8e,7df0feedd81e40f4e96af19fd89d7770]
    POC    4 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1570128 bits [Y 54.1893 dB    U 53.3369 dB    V 53.7289 dB] [ET     5 ] [L0 ] [L1 ] [MD5:23e1dc32c98456ce915bed6865524d69,800e08564a75c98c1e8d5f22421cef56,f0ac5b04af8298053336408c9851a00f]
    POC    5 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1189448 bits [Y 53.9427 dB    U 53.9135 dB    V 53.8718 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ffd964a52918fd295c3702f1e99c331e,40ef3cc607f116e4c874a71d83ef563e,809f497acbdedb31fecfec03cc2d9be6]
    POC    6 TId: 0 ( I-SLICE, nQP 9 QP 9 )    2276880 bits [Y 54.2805 dB    U 52.9692 dB    V 53.0091 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e52b024dd0e8802fd33557ec03965128,b7f8dcaa25a4c42866294d252842cbc6,ddb5aaa531f77c96de7919bc80a6e687]
    POC    7 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1115920 bits [Y 53.5640 dB    U 53.5223 dB    V 54.1760 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c06bb9db75a5f78c723675a2fe3b50c7,6652ff967f3aeb9426083562d79294a0,90992df46d86fc5555d9da2ee9cc31ec]
    POC    8 TId: 0 ( I-SLICE, nQP 9 QP 9 )     999928 bits [Y 53.6202 dB    U 53.8461 dB    V 54.2542 dB] [ET     5 ] [L0 ] [L1 ] [MD5:87ca5cd2a46d4c0e938c6a9ff46a40fb,e7595799b3db1f178e2e316b4c126138,2c3802701033a024149fb33132187d2c]
    POC    9 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1330752 bits [Y 54.0170 dB    U 53.4333 dB    V 53.5371 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4afb01a83f58da97aeb5cce6f60c314c,cb100df1f1ead458ea483867bb92de7c,3fec0fba49dbd1495afb1433e312304c]
    POC   10 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1314800 bits [Y 53.9241 dB    U 53.6197 dB    V 53.5610 dB] [ET     5 ] [L0 ] [L1 ] [MD5:83e2ca7224713e7c21561ff39ac0d991,5d09fd14375bb13e073b8683b2ac5caf,098e6747c275ae57f0e1619b1ba6edcf]
    POC   11 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1388392 bits [Y 54.0095 dB    U 53.4647 dB    V 52.9361 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f49b22bd7c0ad6411f09d93bc766023d,8c0fd3c3f0822078dbe29e5fe363f015,3aac4703db629b069aefa7d553db0bd6]
    POC   12 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1774552 bits [Y 54.0775 dB    U 53.3133 dB    V 53.2550 dB] [ET     5 ] [L0 ] [L1 ] [MD5:452d49ca1060db7ab0a751faa3cef0a3,0313534ae704e83c5712e0320d0beade,44100433284158f25fe31c5de3c8e38d]
    POC   13 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1844992 bits [Y 54.0545 dB    U 53.2716 dB    V 53.1857 dB] [ET     5 ] [L0 ] [L1 ] [MD5:76aa706124d6c0c3959a812869214a63,3c88b0f10cbc92292c1190f5254b1d19,8e81e7ff14ec1f4ebc36065b4f358786]
    POC   14 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1241056 bits [Y 53.8808 dB    U 53.7025 dB    V 53.6314 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d5f70122b1cae5021d365b6c7ef4b028,f7beba1757792cd81cb210730f47f3d6,eb5858ec7c870183d606c4eba53a0aa0]
    POC   15 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1632064 bits [Y 53.9879 dB    U 53.6744 dB    V 53.5085 dB] [ET     5 ] [L0 ] [L1 ] [MD5:196e4de202bc0b78e785a071469cccea,3b9c09e11995e1e39b93c35ab27f87f7,d0db5d3f50a8695566312ce67b73b793]
    POC   16 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1897688 bits [Y 54.1240 dB    U 52.8795 dB    V 52.8892 dB] [ET     5 ] [L0 ] [L1 ] [MD5:31ae69b5ac3ac70706eae4ec45194e6c,d9254b317724a6d24de74580032e5865,2eb33d3047d254ae05fcd4e6792fffbe]
    POC   17 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1526512 bits [Y 53.9449 dB    U 52.9128 dB    V 53.2407 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6076ad98a2d54c8ee72f5d5767d80cd1,20a8f1ca63f0bda3d21a922b9bd2e083,947a3f9f299c2fc7640b939c733c1abb]
    POC   18 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1648888 bits [Y 53.9999 dB    U 52.8479 dB    V 53.0900 dB] [ET     5 ] [L0 ] [L1 ] [MD5:eef9f55dc665c10c3964c922532e8d1d,d5511131931d701a355185e4921b2076,3bbbeee1af49fa6adaca3d662627db65]
    POC   19 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1101224 bits [Y 53.7419 dB    U 53.3942 dB    V 53.5123 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e28062321b489fe39a7706c46cc3904a,8c72187f91e1451c14a3aa1a424fe40b,6c0ae9788623dc49887a0dc74e4b171f]
    POC   20 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1246120 bits [Y 54.0214 dB    U 53.3464 dB    V 53.5736 dB] [ET     5 ] [L0 ] [L1 ] [MD5:5c7fbf21a0c28af8934a110e13e13d89,9fdb1b27cc9585cc31f3cebdbf579f14,2133d6a8b647a30d50ed7d044583ef4f]
    POC   21 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1570352 bits [Y 54.0836 dB    U 52.8773 dB    V 53.2252 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4cf7acc893b825d8e926e34c55c6b5c7,684323d0b910192fb93ce13c801ef20f,d777bac1857d272f5e54e342f12fd553]
    POC   22 TId: 0 ( I-SLICE, nQP 9 QP 9 )     996256 bits [Y 54.8433 dB    U 53.4441 dB    V 54.3657 dB] [ET     4 ] [L0 ] [L1 ] [MD5:6b8be167395049057e13641e709abebc,97457b5aee3a6a5b6b3093a2acb7258b,212c196ff8053a1c4fec8ffb467ea7e3]
    POC   23 TId: 0 ( I-SLICE, nQP 9 QP 9 )    1929112 bits [Y 53.8977 dB    U 52.8778 dB    V 52.9901 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f535a4bb8eba10ea71af8a8bd2e8de41,372d1ecf1504ee9746dc81b81c559e9a,a2afccb279681462f03fc6d737dad647]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   89099.5000   53.9912   53.3348   53.4537   53.7733  

# 8
    POC    0 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1617248 bits [Y 55.1266 dB    U 54.3688 dB    V 54.1723 dB] [ET     6 ] [L0 ] [L1 ] [MD5:768cf359f0c8eaa21df968bdb9364c42,a7585906bc523425cd52364b9b31a1e9,3bb2c9e74af8e6417bc78c2a02346a4f]
    POC    1 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1972432 bits [Y 54.8778 dB    U 53.8596 dB    V 53.8624 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1848fcd6603533345f6d55cfb8c5eb18,9d4b583fb4b076104f92d8f8ddb4084e,f2d707ae96d02e3bf75fd8aff8dc6db3]
    POC    2 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1496800 bits [Y 55.1143 dB    U 54.0222 dB    V 54.2533 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6e89c09cd19bfa39e428785cea50f7a5,ebb375ea60567a083a6dd57c0b127448,a8d95a99d49ac54a5a0909615c6b6602]
    POC    3 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1389896 bits [Y 55.1487 dB    U 53.8961 dB    V 53.9029 dB] [ET     5 ] [L0 ] [L1 ] [MD5:381473f7d1be98ed375f58609fc15ed4,17578f6bb43b6857529c402a36ef72a2,b822f7c518059deb523d89fb3c3dec63]
    POC    4 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1666800 bits [Y 55.2935 dB    U 54.0324 dB    V 54.4091 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f12270d555f675465b2def0810b42d3e,cb920c0a7485371a1826e0a7c2cef6ef,dad5aa4ed14cef66953e9dfedac03ebb]
    POC    5 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1288144 bits [Y 55.0940 dB    U 54.4711 dB    V 54.4358 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ec8f1e817411ea73542dfccd626d6df6,acbb1210dfebd85fcca285d042d1c882,6e8c63363216d32b16fe971b86a55b68]
    POC    6 TId: 0 ( I-SLICE, nQP 8 QP 8 )    2389992 bits [Y 55.4618 dB    U 53.8520 dB    V 53.7951 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f7da74d0d5acae0eed6ecc462d66b892,85c9531951d75876e379ba229af10da3,8f8fca2112d8b5da1d32dbaa02ebf543]
    POC    7 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1223248 bits [Y 54.7041 dB    U 54.1428 dB    V 54.8100 dB] [ET     4 ] [L0 ] [L1 ] [MD5:757195e43eed0c5251cd09475fdca1c6,9aa5cc17046be93c657c55f739006bc7,9d834065e1d9b080de9848aa625fd0ea]
    POC    8 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1107144 bits [Y 54.7637 dB    U 54.4121 dB    V 54.8022 dB] [ET     4 ] [L0 ] [L1 ] [MD5:b6f38d0f455390b0208e744eda016685,569b6934a4915a1760ebadefb263189e,bf1ccdc5f4b63dd0d2f87deeaef92b3a]
    POC    9 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1434208 bits [Y 55.1792 dB    U 54.1366 dB    V 54.1024 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d16b26243b1a877125f14c0a50a98dff,33d0880ca1d89f76562e0c070b833c17,6db8b0ad7b286b1a6001516738a40da6]
    POC   10 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1416440 bits [Y 55.0849 dB    U 54.2813 dB    V 54.1074 dB] [ET     6 ] [L0 ] [L1 ] [MD5:1e337ce6cbf0acf7e3116c695b248917,3fb7fd52cc8d8337788896cf6631b5b7,3ad4d83b25238fd8b7fc97e41b76706d]
    POC   11 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1494928 bits [Y 55.1717 dB    U 54.1599 dB    V 53.7163 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9071e3cc8fc08d5cc0f81edb76321b1f,fec9c7964cd9004411310e009c973509,aa7c32ef101990be567cda817cfbe385]
    POC   12 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1880080 bits [Y 55.2443 dB    U 54.0334 dB    V 53.9113 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e012a225b8997159c7f123ceecc2fda1,f87650cc06cde56c1e5875fb07aebb04,f9f5b591e21a8914766e2d01be9f0daf]
    POC   13 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1950368 bits [Y 55.2023 dB    U 53.9348 dB    V 53.8560 dB] [ET     6 ] [L0 ] [L1 ] [MD5:f575036740e7b4ebaadad27d5c57fe3e,7f23691154e47c17e80edb9c2e358e61,3cb84f5995200c111c4583e10345e0b4]
    POC   14 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1345752 bits [Y 55.0315 dB    U 54.3224 dB    V 54.3114 dB] [ET     5 ] [L0 ] [L1 ] [MD5:571dcc602ee3612b119d26a5352eb235,0e57e28326a9a10c0f48e3e548ace609,b7449d31d855feddec8115aba9ffbd42]
    POC   15 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1736168 bits [Y 55.1363 dB    U 54.4202 dB    V 54.2254 dB] [ET     5 ] [L0 ] [L1 ] [MD5:642660a5c1f15b282f6def7a1f5aa73f,0260449af2f84f91f09ddd4752cea634,84855daf94ddcef64bab37f7b7c29920]
    POC   16 TId: 0 ( I-SLICE, nQP 8 QP 8 )    2010984 bits [Y 55.3065 dB    U 53.7864 dB    V 53.7011 dB] [ET     6 ] [L0 ] [L1 ] [MD5:81b905970df5b1e3faaff422fb1899ba,e0dbeca83f8c657becb271f1df5a9d4d,a7247800b2b7e51ae3099705b110e415]
    POC   17 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1630216 bits [Y 55.0789 dB    U 53.6481 dB    V 53.9069 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f018c81eec7e79a128bce8dba54e4e5a,3068085f7093679aed25acdbf2428015,18e674aba6f21bf34d488c6bc36559b8]
    POC   18 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1762920 bits [Y 55.1845 dB    U 53.7332 dB    V 53.8288 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ffe93aac18692ea2ff083e54603122c2,28afc6bc3249f3398b9c01a28b53cef7,945c887abcf9a493f362d95ecb9b37c0]
    POC   19 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1210224 bits [Y 54.9310 dB    U 54.0398 dB    V 54.1208 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7c83e6690ce04e827316da27b89f8967,5d27bee4417e644f880d03a9cba4f16f,5abaac20519aae200f9772a800324f6b]
    POC   20 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1349960 bits [Y 55.2083 dB    U 53.9098 dB    V 54.1266 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e86ac8c12ef9bee16273a6e4648d169e,eb03b8fd85a5e1c98172224f647cf8b4,197c7fd460ba1a743246b92fbf1c3bc6]
    POC   21 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1679872 bits [Y 55.2627 dB    U 53.6960 dB    V 53.8622 dB] [ET     5 ] [L0 ] [L1 ] [MD5:794dfc997a986049d5af4feee69245d9,db1802349243cd2f78c3902c4bd7d0cd,f48f275be243fe86df21e7acf90456ce]
    POC   22 TId: 0 ( I-SLICE, nQP 8 QP 8 )    1080688 bits [Y 55.8035 dB    U 54.3457 dB    V 55.0713 dB] [ET     4 ] [L0 ] [L1 ] [MD5:4f4094dfed902da91138cda8fb0a462d,dc68c95aeb6a82fcf9c1a10960cfd21d,9ae277bb7d219cfaf18171c441d3d03a]
    POC   23 TId: 0 ( I-SLICE, nQP 8 QP 8 )    2039224 bits [Y 55.0417 dB    U 53.6363 dB    V 53.7527 dB] [ET     6 ] [L0 ] [L1 ] [MD5:f4766f090b664a417450ff4509d01059,c9a71743a95fa6c789b5e20d0fe1c83d,3f90b725536b33a1ee4e4a403a9ae82c]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a   95434.3400   55.1438   54.0475   54.1268   54.7548  

# 7
    POC    0 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1765272 bits [Y 57.1097 dB    U 55.5200 dB    V 55.5239 dB] [ET     6 ] [L0 ] [L1 ] [MD5:c97013ffe6cbcfe1ce7af972213e489a,fce143e784acfe758adcc399dbe9c187,08dd06df9167de7c8f1900d7c36281a7]
    POC    1 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2108920 bits [Y 56.4452 dB    U 54.9713 dB    V 55.0172 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5a6390a326934f91397e31e3029cf0e9,69e6ba9d3a5c6a99da3e930c5f1165a6,a377652f2e3baa414d09c5ab42a11cd7]
    POC    2 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1645576 bits [Y 57.0117 dB    U 55.2720 dB    V 55.4422 dB] [ET     5 ] [L0 ] [L1 ] [MD5:406f38a4571fae32f180d97d2b26f989,1ca1f38624fa827ff6d3f31b94fe622a,853ed62e587dcddb17d62ef8b6cb4437]
    POC    3 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1555560 bits [Y 57.0061 dB    U 55.5638 dB    V 55.6137 dB] [ET     5 ] [L0 ] [L1 ] [MD5:df99a8a7ecd07ac712101ca2f94cfe01,9c96373b844703eaddf5068e24cbb1c9,4cd0198b3df6778d39eb016b295bed24]
    POC    4 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1807912 bits [Y 57.0902 dB    U 55.3221 dB    V 55.4913 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7a71a1ceabcb6b68281933bd003253fd,c66830399d0f16587d85e4432ea1572f,744edefb0094ce98ae0b095d87d252ed]
    POC    5 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1445720 bits [Y 57.2288 dB    U 55.7250 dB    V 55.8398 dB] [ET     5 ] [L0 ] [L1 ] [MD5:de4106b631436fb22dfaa8b192b49b76,ad1a599748a4b8da61b68a2e917ce9f7,a7d43b80d0eda634fb9242a7c8f8dd9e]
    POC    6 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2528496 bits [Y 57.0862 dB    U 55.0885 dB    V 55.1015 dB] [ET     5 ] [L0 ] [L1 ] [MD5:405180321ae1166e0b6b89479e6c7f22,ac84a121cfbe9d0a984f27106e039049,17385043133e368ec83f1f779029701e]
    POC    7 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1404336 bits [Y 56.9728 dB    U 55.9189 dB    V 56.2863 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7393b72865a470d0ef98b73f2305db55,689d6fe315c7851f763791dac9796867,ff795ac73e62d76d77d7c8b44c64aa18]
    POC    8 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1300584 bits [Y 57.3116 dB    U 56.0606 dB    V 56.4947 dB] [ET     5 ] [L0 ] [L1 ] [MD5:de483ba1edd341c87a14c007a204e18d,b9f8925ad0d4d28eb7495bbfde2dda00,4f7c33088c8705cffc330483cdc706c4]
    POC    9 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1581760 bits [Y 57.0136 dB    U 55.3714 dB    V 55.3584 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9d66e660dd71308a066e0f1a211d0449,b06a2a970bd5b8b8593bede0f7fab636,85209df419c53793091b4d7f4770a5e3]
    POC   10 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1580040 bits [Y 57.1983 dB    U 55.5621 dB    V 55.4934 dB] [ET     5 ] [L0 ] [L1 ] [MD5:2cf7d196937798ac16fe0614e5bec970,10439f222b64c99b0abf1f3f126a452b,b47c17432ee97356635dbca8bc1f0b7d]
    POC   11 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1656016 bits [Y 57.3164 dB    U 55.4358 dB    V 55.1778 dB] [ET     5 ] [L0 ] [L1 ] [MD5:84d8df2a3114a5b24b7d0714a006426b,fc3913d37f1db9acffab31c1ef640e1c,3aaa47390b70af1e46068bb6764798af]
    POC   12 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2008920 bits [Y 56.8635 dB    U 55.1015 dB    V 54.9801 dB] [ET     5 ] [L0 ] [L1 ] [MD5:eaa8ba6482f24cf979551157f457f1d5,6505f6a6b507af811fa52369a3b21b99,faf965d224d786e982ad831bcb895981]
    POC   13 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2088928 bits [Y 56.9176 dB    U 55.0776 dB    V 55.0095 dB] [ET     6 ] [L0 ] [L1 ] [MD5:136ee2ee71f68cc0c41d6462d51fad1b,e4028154e829f0a4cc91bf990eceaea8,7e398a26c367765606336b41024d1607]
    POC   14 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1535088 bits [Y 57.5572 dB    U 55.9654 dB    V 56.0653 dB] [ET     5 ] [L0 ] [L1 ] [MD5:fb56ddb653579856037b7c2769480681,f12e8b973110fc1052c2958e432e7c2a,f763543807d1b559135155c07ecfb37a]
    POC   15 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1881928 bits [Y 56.9932 dB    U 55.5844 dB    V 55.5002 dB] [ET     6 ] [L0 ] [L1 ] [MD5:05bd384e1a03eba95c530c225c8127f0,0b39d4466d9093eaf7b4f472f176dc35,c913043c6f940688a617829cdb720ea0]
    POC   16 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2161408 bits [Y 57.0073 dB    U 55.2669 dB    V 55.2289 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bbb1ea0b85cd472c121f57af71f96a14,2069a4e84cd53d48625fcfdbd29d76a5,837fad8df735f58d9d077b0e598f08d7]
    POC   17 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1789088 bits [Y 56.9594 dB    U 55.1437 dB    V 55.3154 dB] [ET     6 ] [L0 ] [L1 ] [MD5:47b4af861eed3d40ced7c8b5d0a2f04b,fe29167d507f309c67d4cbca5feebe17,54e5f688553e66a8687b4864a2e93a73]
    POC   18 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1913720 bits [Y 56.9352 dB    U 55.1825 dB    V 55.2742 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7378a10764ea086eb895017fa010cd9e,2253f3d7d1d78f178fcf7386c28eb4e4,efbf1eb9cd06c0125cbc578ac70ad920]
    POC   19 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1387288 bits [Y 57.2034 dB    U 55.4551 dB    V 55.5487 dB] [ET     5 ] [L0 ] [L1 ] [MD5:5821fa9e5b6a838803ebe03f6bb4b9c3,346c30b7af6d5ffb01f25c9f4dd080a4,72f020b1e38dfce62f44f5836d0c25db]
    POC   20 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1525488 bits [Y 57.2010 dB    U 55.8740 dB    V 55.9376 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9797110c62905979fe81b17c303bf163,497a6e6717a8244520e1c6b539fa4759,1fe6103c1a1b35121f739f3788ac3dd9]
    POC   21 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1855816 bits [Y 57.2274 dB    U 55.6251 dB    V 55.6296 dB] [ET     5 ] [L0 ] [L1 ] [MD5:2e5bbc1659a4b7d6922ea5319fbf80ba,e64911903424b22207f2cf61b2f231de,fa093758163cb265f5a7bf7d7717ac3c]
    POC   22 TId: 0 ( I-SLICE, nQP 7 QP 7 )    1240184 bits [Y 58.0650 dB    U 56.3804 dB    V 56.8111 dB] [ET     4 ] [L0 ] [L1 ] [MD5:470c3e7f86e4b530cae1b7d691e6973f,f8cd1044a02484622eb8c92a623ea7fb,f02da91203ff6788c60bd5a871f32402]
    POC   23 TId: 0 ( I-SLICE, nQP 7 QP 7 )    2182400 bits [Y 56.6509 dB    U 54.9694 dB    V 54.9835 dB] [ET     5 ] [L0 ] [L1 ] [MD5:454793d2ec481436b66bf07bc9eb19db,7f2a9957c779510df9bdb60b09fb2a87,709e66c88d2500b309da1335f44c9e13]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  104876.1200   57.0988   55.4766   55.5468   56.4888  

# 6
    POC    0 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1898960 bits [Y 59.1463 dB    U 57.1723 dB    V 57.1727 dB] [ET     6 ] [L0 ] [L1 ] [MD5:490a3c94f9d26988c82f1e379fe1a513,c9f75c8e20f4ec894c4f35983a39b170,be519990a751d546d6c3f195cab9dd4e]
    POC    1 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2246064 bits [Y 58.3774 dB    U 56.4500 dB    V 56.4488 dB] [ET     6 ] [L0 ] [L1 ] [MD5:c3430cd4b234a6de210158eaabaf42d7,87d16d7d54d92a960444fd0d347c8aec,6251467ce033cf9ca1b6e93bd61c5c40]
    POC    2 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1797880 bits [Y 59.1381 dB    U 57.4620 dB    V 57.3721 dB] [ET     6 ] [L0 ] [L1 ] [MD5:aba85c5c5a803365e2d65f598ebe4aca,5117a4d2266f12730d2ebd22e9df3cd7,220bfa0dec0ece4cf1f7f2a6285da7fd]
    POC    3 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1701448 bits [Y 59.1407 dB    U 57.6331 dB    V 57.7473 dB] [ET     6 ] [L0 ] [L1 ] [MD5:0b90da9af32ff18af240225b547dc6ca,b43821cd6aa94afeba26f979d149597d,8cda286afb142dc108e3ba3041665efc]
    POC    4 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1943744 bits [Y 59.0569 dB    U 57.1031 dB    V 57.1483 dB] [ET     5 ] [L0 ] [L1 ] [MD5:dc51f23a7a8bfc69332936adf5e52bb6,eb350389a938f31313821941a4b7b4b0,73d0a309a2e2428a266660f13daca21f]
    POC    5 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1587280 bits [Y 59.3583 dB    U 57.7106 dB    V 57.8687 dB] [ET     6 ] [L0 ] [L1 ] [MD5:8c0b9c4aba87b30f9507524198bfe2ff,2a82e9a9093065f6df8c546d45b948ca,7018c6372f88fdad0adc4c7229505cea]
    POC    6 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2668176 bits [Y 59.1195 dB    U 56.6801 dB    V 56.8370 dB] [ET     6 ] [L0 ] [L1 ] [MD5:973855070a68d7ab139604682ed90650,c4fbe928ad3518aebda454534f8fe33b,055b7f112836a2d63dc922f5eefefafd]
    POC    7 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1541296 bits [Y 58.9758 dB    U 57.9084 dB    V 58.2306 dB] [ET     5 ] [L0 ] [L1 ] [MD5:cbab17a228ffa4029f6abf7c2ba40bb1,152cae8722e3cd1a994123e65daef11a,9e280f0f93da2386cc7eb80a7afd3389]
    POC    8 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1437112 bits [Y 59.3510 dB    U 58.3252 dB    V 58.7644 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3a0b61bdcaebcf3a87232d6cf8b293cb,68e1c56ad8cae51842ae6306f5a7ab91,f9a047760d1fa0078ae4b9e792def701]
    POC    9 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1732360 bits [Y 59.0761 dB    U 57.4209 dB    V 57.6544 dB] [ET     5 ] [L0 ] [L1 ] [MD5:8d18cf85de23422eeb59017efc8d187b,4acbf67f70ac1539966ce51bebde525c,3141c76df35732807a45705f8d76d8a5]
    POC   10 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1714800 bits [Y 59.2012 dB    U 57.3832 dB    V 57.4164 dB] [ET     5 ] [L0 ] [L1 ] [MD5:059c3b60589c3c52af702bf78072217f,ff74483111e3d14a579e22837fb9ec6d,b3a1af8fed07a21d32164ce00aa5fdb0]
    POC   11 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1800200 bits [Y 59.5122 dB    U 57.5184 dB    V 56.9756 dB] [ET     5 ] [L0 ] [L1 ] [MD5:95e5ee57cfb3a7cd00ed9547fd2195e7,cfe40e9eace1a97e03aca60e54e6c30c,ed91435718dd86846a80c35433d06bbf]
    POC   12 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2149504 bits [Y 58.8813 dB    U 56.6543 dB    V 56.7198 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e3bf0e629014b2e538c3351310b9a752,880df1bec76ec02c7cb98006ee84f3b6,7f34d4f78644af45784f8f336bcd0743]
    POC   13 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2234896 bits [Y 58.9635 dB    U 56.9510 dB    V 56.7677 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e8c77cf99295363166939c3325328205,3e1a872aeeb7a5a09ff948ca80a74208,96342d167afa55c4c1863a630e36e673]
    POC   14 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1674304 bits [Y 59.7332 dB    U 58.2023 dB    V 58.2701 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6ea8e4bf18bc521edffcd70c8da91445,2ee77925e3fe94a459403f124c6ed683,8810cc73fdd86aeb4d84ea41cc725e2d]
    POC   15 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2022984 bits [Y 59.0442 dB    U 57.4172 dB    V 57.4149 dB] [ET     6 ] [L0 ] [L1 ] [MD5:403d1d65a04aedd7a4c0f517468eb6fa,1ea30b8398f750ad695db801a7504c25,2dc32d2b99f81cc8c5d30ab165ccadb3]
    POC   16 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2296640 bits [Y 59.0364 dB    U 56.9994 dB    V 56.9922 dB] [ET     6 ] [L0 ] [L1 ] [MD5:1bdc6c5a0e8792542375ea1ba4b2b66f,c5bfac6e50fb9c1e9e3ee3421c99b3a7,d16740e3f0d60fc348aa57d8cbc864c0]
    POC   17 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1936488 bits [Y 59.0568 dB    U 57.0722 dB    V 57.1979 dB] [ET     6 ] [L0 ] [L1 ] [MD5:4b21a895ef76aba43705ba90ea74fe1e,b027a67131faa5d52241b19b8f1eacef,8b3cd177ad2857419194a4d61d0bb29b]
    POC   18 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2049728 bits [Y 59.0218 dB    U 56.8115 dB    V 56.8725 dB] [ET     6 ] [L0 ] [L1 ] [MD5:2be4106906aca767218b7b6d355ebb76,4a75cfbe3d89172490cc507f3c9909cd,b664e8fbd540bfff3c4a87ec6ae4df7c]
    POC   19 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1536664 bits [Y 59.3056 dB    U 57.4848 dB    V 57.7875 dB] [ET     5 ] [L0 ] [L1 ] [MD5:560a4d22eadc6810e29238e53c2d2709,b03b799dfecc40684ec483162096e4e3,5133dbb111aea73a697ce2d6536b03d2]
    POC   20 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1667888 bits [Y 59.3145 dB    U 58.1812 dB    V 58.2711 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6dc1d98c1a205206e51d846e35bc9180,d28d048a9a5537e2e8b98d8b1d57fdc0,06ba3c109912235f6e6dd35a12895004]
    POC   21 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1984096 bits [Y 59.2968 dB    U 57.2905 dB    V 57.4628 dB] [ET     6 ] [L0 ] [L1 ] [MD5:af0495dd89bbcef9d7b9320fbc16882e,65f682c631c0a49be9584c7c11ad8bf4,6161e69eed54732fd0ac822642165425]
    POC   22 TId: 0 ( I-SLICE, nQP 6 QP 6 )    1366840 bits [Y 60.7451 dB    U 58.1669 dB    V 58.7369 dB] [ET     5 ] [L0 ] [L1 ] [MD5:27846435f2d9b0bc223b9ea3756dfe24,907de460c95d229ca45b1e4c59ae21b0,070983805f922f5b968a013b54ae3497]
    POC   23 TId: 0 ( I-SLICE, nQP 6 QP 6 )    2325312 bits [Y 58.5829 dB    U 56.7147 dB    V 56.7157 dB] [ET     6 ] [L0 ] [L1 ] [MD5:8c391ccee997f91e587d194a01bc6952,459fb03c2c059e81e418f7aa49d8d455,59666c760ce57748837792240347d827]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  113286.6600   59.1848   57.3631   57.4519   58.4808  

# 5
    POC    0 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2028760 bits [Y 62.4223 dB    U 59.8460 dB    V 59.7474 dB] [ET     6 ] [L0 ] [L1 ] [MD5:c60ae3b1e16ba052d8e006ec5bb86694,b5fd55552b2b042872d0a2227e6dc48c,e8e33135ff6a1a9a58877968cd0c4345]
    POC    1 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2386960 bits [Y 61.0449 dB    U 58.9535 dB    V 58.9572 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e2acd67afd1fc1c9b7ff2d5172ff4db8,a6e5dfa8e5d6af5e9a14c0214d331454,e640c39b2ab4fd6947f62eb215793067]
    POC    2 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1943304 bits [Y 63.1468 dB    U 60.9511 dB    V 60.3731 dB] [ET     6 ] [L0 ] [L1 ] [MD5:fc58452ccc1d6c6e4e78040b7631c30c,b0ebdcb054cc5e47323eade36d7cbef2,10719c375cd0316ce8605f815b2db216]
    POC    3 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1839040 bits [Y 63.2965 dB    U 61.0903 dB    V 61.4403 dB] [ET     6 ] [L0 ] [L1 ] [MD5:60e4a35dcacc982fc74e416e2c1b1e65,71fbc0e8cebdbaaf5c15245efb0f8c32,0044c5c9fdf4bd83ce876c1d3c431a66]
    POC    4 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2070656 bits [Y 62.1797 dB    U 59.6354 dB    V 59.5981 dB] [ET     6 ] [L0 ] [L1 ] [MD5:330a06cbd6ee2c99a5aaeb4c20f87c8c,992245bf035f3702922b04b1ea16fe9c,90e74070bb74a726223de01416e644bd]
    POC    5 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1722736 bits [Y 63.4986 dB    U 60.9893 dB    V 60.9132 dB] [ET     6 ] [L0 ] [L1 ] [MD5:a01f11fb1847e9d899c256bc7e988a50,6ba51ee1f3533850b9ec40f87f51e877,9f212e22675348aa72b030e45fcd1e2c]
    POC    6 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2805632 bits [Y 61.8894 dB    U 59.5335 dB    V 59.9737 dB] [ET     6 ] [L0 ] [L1 ] [MD5:09d6c173c917c14bce521707735cb190,dc82101a35921e7eea527ddac53b9386,0402eeeca6deca15973e49ec1606cdd5]
    POC    7 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1674632 bits [Y 63.1830 dB    U 60.8615 dB    V 61.1017 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d5e5de9e85d033092e2860894a3c9615,34d0bf05b1d175c3fa3ef2c09d5bb332,34ad432004c3f4c4c34a9551af875b81]
    POC    8 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1568288 bits [Y 63.8499 dB    U 61.6428 dB    V 61.8375 dB] [ET     6 ] [L0 ] [L1 ] [MD5:afa68f218f17ffbf6038d4c702816528,15f89dc83574b372a9f8d397d234ddf1,4b26f3373ba5bf66a45620fe44ac66d7]
    POC    9 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1877808 bits [Y 63.1633 dB    U 61.0142 dB    V 61.2009 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f7dfddab5d41518d64668a803a73d182,1f685fd92131ddd45041ce0b6c76ac3c,0e5aa4cbd70c08ae3d5043601e7a1416]
    POC   10 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1848552 bits [Y 62.8298 dB    U 60.1284 dB    V 60.2339 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e4e38d6cf01556f61d14b9821815c942,275bd88414418cddec689ef5f7450726,c41ceafcdb2c706bb80fd75f8afdcf12]
    POC   11 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1949656 bits [Y 64.0001 dB    U 61.2996 dB    V 60.7795 dB] [ET     6 ] [L0 ] [L1 ] [MD5:8f8ed8b8f6c6f9ea7aa312dffa388a4a,4ff6b4b027fbe5442084302fbc833418,e50499a50d9b477975c81149082a55b1]
    POC   12 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2280312 bits [Y 61.7765 dB    U 59.1350 dB    V 59.2182 dB] [ET     6 ] [L0 ] [L1 ] [MD5:1e45a261714d7ae5bf409e2b71480750,9d7bdb2d874dc9fa2b2db34d6e7bc614,52c3ee05cd8ffa786a0a8d1f3afe2e17]
    POC   13 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2381560 bits [Y 61.9940 dB    U 60.2347 dB    V 59.9988 dB] [ET     6 ] [L0 ] [L1 ] [MD5:138c652cd5265ee32ffaf662ffaa5bd4,3e99215b1a13df61b3cba3eb8259ee82,8a293ee93b4ec189984af2ab1c49b279]
    POC   14 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1794480 bits [Y 63.8171 dB    U 61.3070 dB    V 61.1973 dB] [ET     6 ] [L0 ] [L1 ] [MD5:dbf3442409756e85214c8480652dfc71,306fe84a63da192932a82efe8bd8966c,a1eb6a7289c5c263d728cd48f9705a05]
    POC   15 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2160872 bits [Y 62.6121 dB    U 60.4013 dB    V 60.6957 dB] [ET     6 ] [L0 ] [L1 ] [MD5:83f466a9cb54af7939b6da0835252c42,e124d8feadb780c9805f58dd161efbf1,b9f2dc5d5c7e3c93b4a52262c41f8533]
    POC   16 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2424992 bits [Y 62.1570 dB    U 59.6417 dB    V 59.7053 dB] [ET     6 ] [L0 ] [L1 ] [MD5:017337c19ba9bbcd9a13a0d9c298c368,010dd269874a134762ab475fd6bfa8ae,8f7f2252c33f48a093a43494ddb37a06]
    POC   17 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2086464 bits [Y 62.9110 dB    U 61.0279 dB    V 60.9065 dB] [ET     7 ] [L0 ] [L1 ] [MD5:9d25f97dc756d4c6d2821639f33140d9,72c8bed76ef93a2e7130ebd193cc0596,4995c004d192beb685582ca3c95d0d85]
    POC   18 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2187680 bits [Y 62.4721 dB    U 59.5882 dB    V 59.5863 dB] [ET     7 ] [L0 ] [L1 ] [MD5:5b2721e3c648b81476139fc86c26da9f,25f8ba91e08c950aee4529f3e0a3df1a,8ad5bc22b624d14de1563be9f805e409]
    POC   19 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1683528 bits [Y 64.2830 dB    U 60.8359 dB    V 60.9300 dB] [ET     5 ] [L0 ] [L1 ] [MD5:807d8cf98cbed02278c06cabe1c2655c,780bce51554226f3ebc4240503b1e94c,c9fbdee18d141168514db134bc0ae029]
    POC   20 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1791560 bits [Y 63.8384 dB    U 61.2759 dB    V 61.3042 dB] [ET     6 ] [L0 ] [L1 ] [MD5:03745ad620c24a5942e7a5e9f4552fa4,eada4239a5636a43d15c9fe71a20ef26,44b9b0afbcd7cea123992b7e6cd3a491]
    POC   21 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2122296 bits [Y 63.4198 dB    U 60.6003 dB    V 60.4947 dB] [ET     5 ] [L0 ] [L1 ] [MD5:3d9bb86b610b27761303c0f1d4720581,f3d4bd5c60babfc55d65008b71229da2,1528992effad8c16610623b30bc49f62]
    POC   22 TId: 0 ( I-SLICE, nQP 5 QP 5 )    1470968 bits [Y 64.7448 dB    U 62.0642 dB    V 61.8834 dB] [ET     4 ] [L0 ] [L1 ] [MD5:f700e8a3de8901d8f3a30ead4e59f4e7,eb4aa380957a145ae66ca1f57f3724fb,8d53c2c8f78e09e422060550d2945e02]
    POC   23 TId: 0 ( I-SLICE, nQP 5 QP 5 )    2464200 bits [Y 61.2880 dB    U 59.6592 dB    V 59.6649 dB] [ET     5 ] [L0 ] [L1 ] [MD5:489872c575184d2ef192be0b60d92f85,981dafd6cb4f3ad71d110a4ef41b37c1,10be11a6d86966fcc8ed6fd6949cf6cf]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  121412.3400   62.9091   60.4882   60.4892   61.8556  

# 4
    POC    0 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2138808 bits [Y 69.0926 dB    U 63.0254 dB    V 62.9270 dB] [ET     6 ] [L0 ] [L1 ] [MD5:fee047df099360abca0d5ce6e72725dd,dadf878e3959b28f3d3c5f2d4698d449,01961727e9d30cb6bd955a4c1bd7b627]
    POC    1 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2515296 bits [Y 66.3116 dB    U 61.9670 dB    V 61.8127 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7f701576d76e4f7184bd5fbd60aae170,8f0db07e5dcfd826b9842d3d36823012,103ca3afed152a2ac44b4f8e4e169ef4]
    POC    2 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2029416 bits [Y 70.1291 dB    U 63.7509 dB    V 63.2940 dB] [ET     6 ] [L0 ] [L1 ] [MD5:003b68cfcb07f648ccdcd344fb330a33,ac7dea52b962fa5d83e4975027740b44,76b0b4ae9acfe1f52be8f85c0fd89a60]
    POC    3 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1912384 bits [Y 69.6822 dB    U 63.7252 dB    V 63.8061 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5c6f556c9a3a6ecb211206a7ac14d444,01eb83787bafad9966bd89ba47680f4a,fe6c45f2a5212e06e9008d739238646b]
    POC    4 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2177976 bits [Y 67.8024 dB    U 62.5923 dB    V 62.4802 dB] [ET     6 ] [L0 ] [L1 ] [MD5:16a8ae687b8a314b05275c37fe77c735,7513438092f98073df501467d409f694,f454e45795d1abd559a20369a3b5a9f1]
    POC    5 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1801528 bits [Y 69.8776 dB    U 63.6854 dB    V 63.6537 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e21268803fcc0340262dd42c36b5e63d,06420d899401e1ac1ece647160b8b46c,b36e548ee8805ea79214b61bd983860c]
    POC    6 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2931728 bits [Y 69.4084 dB    U 63.4355 dB    V 63.7060 dB] [ET     6 ] [L0 ] [L1 ] [MD5:969544f6de4174c98f57869282174ee9,763a8c543f7281d96cc951d371530a8e,e63f700a8ac947ccb8c9abf517f34cfa]
    POC    7 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1750168 bits [Y 67.8137 dB    U 63.2249 dB    V 63.2752 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c46b100e277456415c7dfb529ef06a93,161ab0c065ad3dcf0b9656437639b20b,c80173df74ceefbd69498f5e88544e89]
    POC    8 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1629320 bits [Y 69.1286 dB    U 63.8487 dB    V 64.0789 dB] [ET     6 ] [L0 ] [L1 ] [MD5:db91164059298f3c40023b0aa60fd650,6ce87d5e4aea03af722d3ec302ce8a70,04bb76934c7b7b76adb069bc00b44ac0]
    POC    9 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1959248 bits [Y 69.9107 dB    U 64.1472 dB    V 64.0980 dB] [ET     6 ] [L0 ] [L1 ] [MD5:08f8c6c849c8b1bcf418a5c86f00b735,7a8f7029eb544391332dcb1927460b95,a526e0e2d6eeddacaa63cd022bf061c9]
    POC   10 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1941904 bits [Y 68.7662 dB    U 62.8978 dB    V 63.0281 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1ba1305342a48c81aae9b923aef7e6aa,b068e854e22ecd33e40633740f6f1480,a547622916740c884b996e027b34698b]
    POC   11 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2020552 bits [Y 71.8240 dB    U 63.6396 dB    V 63.4822 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bd10c25d475212c51d222c815fa52412,eefe7ef7c4a3655ade2303aa5e90f1e3,6ea425f5a4b650381ace1660b30ab063]
    POC   12 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2404312 bits [Y 68.2440 dB    U 62.2143 dB    V 62.3464 dB] [ET     6 ] [L0 ] [L1 ] [MD5:6b598d98d45d2c81a53c65275ebae983,96eb79424b2ed7823dd5299bd70b794d,979328ab6b0f179d2dd590d7e880458d]
    POC   13 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2486024 bits [Y 68.4625 dB    U 63.3480 dB    V 63.3071 dB] [ET     5 ] [L0 ] [L1 ] [MD5:0d337bd4e9c02be8bbabbcdfee84f263,72db0f6022b8f20325d6c926a32858ed,299f3a309d23c55082305fd58dadb68e]
    POC   14 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1867352 bits [Y 70.4788 dB    U 64.0719 dB    V 63.8273 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4ff2c54c33b628b964af1b9f0fff8e76,ac2c842037eafe9696d6df967c85fa67,5c945ee56a07f84dc01fbe8a6e685312]
    POC   15 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2248976 bits [Y 68.6043 dB    U 63.2868 dB    V 63.5958 dB] [ET     6 ] [L0 ] [L1 ] [MD5:569748c9a5fd5aecd3205f58076605a2,13ca1b0286d67a80e2675b813058f557,6a3612c980bacad3e0f3ee412345f40e]
    POC   16 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2537376 bits [Y 69.3936 dB    U 62.7699 dB    V 62.8464 dB] [ET     6 ] [L0 ] [L1 ] [MD5:948eae223da1ed94df23ceba0e78e324,21df5962bfcdedc98161e757c12f88a3,d6b1c0e9c15128cd949f02c2dbe800b9]
    POC   17 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2170480 bits [Y 69.6040 dB    U 63.9269 dB    V 63.7687 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4082299fa7ace32911de833c65cbd4dd,418cfaa68214dae1d2f27a5256ff61ab,1eab88fd56a4071ba3028aed3c781713]
    POC   18 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2296640 bits [Y 69.4442 dB    U 62.6871 dB    V 62.6134 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bd89ffd30a30db55420a94c86fcdea34,f1aff65f3a15e7c7776e7fb596962ab0,fe150a23cc7606a9965278aac27724f8]
    POC   19 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1751256 bits [Y 70.3812 dB    U 63.0583 dB    V 63.2868 dB] [ET     6 ] [L0 ] [L1 ] [MD5:35bc143804d6273ec182248382f29843,1044597e7161694c9efd5d4b4189b1e8,0a6089e79d99113f6a0909bfcc834979]
    POC   20 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1857256 bits [Y 69.8875 dB    U 63.7108 dB    V 63.9572 dB] [ET     5 ] [L0 ] [L1 ] [MD5:957ff653f803390ee1e091b63a04d91c,dc3ca21df338d5139f17c4822654f44b,8d29e0a5c0aa8dad63c7003c9a278fe6]
    POC   21 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2206872 bits [Y 70.3812 dB    U 63.4792 dB    V 63.4280 dB] [ET     5 ] [L0 ] [L1 ] [MD5:1cdcb15b2eb17993f5a797b1c2cf5f71,fca851b309897d6b810709146c916826,1faf30a07a8058e1c8d750db913d3c56]
    POC   22 TId: 0 ( I-SLICE, nQP 4 QP 4 )    1528400 bits [Y 70.4883 dB    U 64.7220 dB    V 64.2148 dB] [ET     4 ] [L0 ] [L1 ] [MD5:56dec14a8876bbbfccf3544d547036f3,c7191c7f6d3deae8f78c32c502971a1a,6f2652460d47c51dbf3574b1c5ffc49a]
    POC   23 TId: 0 ( I-SLICE, nQP 4 QP 4 )    2573224 bits [Y 66.9464 dB    U 63.0241 dB    V 62.9230 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6b97d0a4a4f72b712921ac35c73e3963,5086f2c4357f985db802bcd6cf2d54f6,ec33a870879bad07352a0412868f8ab1]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  126841.2400   69.2526   63.3433   63.3232   66.2207  

# 3
    POC    0 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2178000 bits [Y 75.5280 dB    U 64.3718 dB    V 64.9157 dB] [ET     6 ] [L0 ] [L1 ] [MD5:cc0f21080c5e6eb5f440dea40b209d3f,1326aef611f94ae0518d216314ed50ea,4c13e1872810daaa9e9794cf0e7b21a6]
    POC    1 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2579584 bits [Y 72.6626 dB    U 64.3904 dB    V 64.3366 dB] [ET     6 ] [L0 ] [L1 ] [MD5:34dff69dc00ef6419fca2dbbd81ef1fa,7e519340c193a7cb4dac7c009a02c89d,19d85afbb39ac8c7953d6823416d99a9]
    POC    2 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2062840 bits [Y 76.3101 dB    U 65.0506 dB    V 65.6610 dB] [ET     6 ] [L0 ] [L1 ] [MD5:502611801fee7fd30e71299ace13b5d3,efc8092eac59b5e08456ee6580481f30,3562a388e8e205a2da8b657f4df817be]
    POC    3 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1945392 bits [Y 76.4956 dB    U 65.8949 dB    V 65.8112 dB] [ET     7 ] [L0 ] [L1 ] [MD5:f6cd750ee89f34ad2a187c5122b84aa5,766f0db851578d44a125600c6dda1d47,6d55f5ce05a7628af7619bef626c3c37]
    POC    4 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2226224 bits [Y 74.2771 dB    U 64.4620 dB    V 64.3923 dB] [ET     6 ] [L0 ] [L1 ] [MD5:2d970be4f26af78530de69428a21b4ed,b13ce8f97dfcd138bf731e955ed26bc0,a6ed9ae13e4fa9a004f2e836babba939]
    POC    5 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1835664 bits [Y 76.4956 dB    U 65.1875 dB    V 64.9816 dB] [ET     6 ] [L0 ] [L1 ] [MD5:79f4f777508b98e3bfe8a87a5c58f478,92b7c23671a37679f71bf943ce88b5a3,54eb04c6bc5cbd4238e4fcfbcbdd87d8]
    POC    6 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2988728 bits [Y 76.5798 dB    U 66.3422 dB    V 65.9213 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e4fda4064b1925174770452b0bdb90ba,72d21972c0511a7cc5ec8d3d34d9d5e4,d835259c88d96b0b070289abcc9b39da]
    POC    7 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1793584 bits [Y 73.2135 dB    U 65.5038 dB    V 65.3942 dB] [ET     5 ] [L0 ] [L1 ] [MD5:5590e9157dd4ccba9fd422c0bcc7c774,8711dd8893d834dd61312de66cfea863,92f0d2aa14104037f3128992f4625c3e]
    POC    8 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1667912 bits [Y 75.3149 dB    U 65.5743 dB    V 65.7215 dB] [ET     5 ] [L0 ] [L1 ] [MD5:955e9f2da4602de0be9876e1d42095e0,4f726b7f487932826eb4d8555872ffd6,1df2842addcf9960990782ebe6a50ec2]
    POC    9 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1994480 bits [Y 77.3654 dB    U 65.8372 dB    V 65.6937 dB] [ET     6 ] [L0 ] [L1 ] [MD5:8afe834e78daa5ee6f261687904cacfe,b530a3b4ccf242df76fce40391d6054c,a284348bfd4ff3353fdad0f8923f5872]
    POC   10 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1983536 bits [Y 75.8034 dB    U 64.4772 dB    V 64.4525 dB] [ET     6 ] [L0 ] [L1 ] [MD5:2339f3943f75e0def44987c3328b5c48,fcf4458f48046472f0035f23294c1167,4be3165d4830e787adad49ee16fa9cd4]
    POC   11 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2056776 bits [Y 79.5901 dB    U 65.3684 dB    V 66.0920 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7ddacfe5254078499b285cefff5e3e26,0b9c58ae79eaa47c75192c5c4d4eadd6,ffb693dca813904df3e2ea3a85967638]
    POC   12 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2453728 bits [Y 75.3034 dB    U 64.1472 dB    V 64.1666 dB] [ET     6 ] [L0 ] [L1 ] [MD5:b126c2f6731eb26772ff0234e8133e46,d5152c45f4268b96e5b075d97b25fcc7,ca11260392ea24443eb5cdf9c50ebab1]
    POC   13 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2537872 bits [Y 75.0571 dB    U 64.6087 dB    V 64.5774 dB] [ET     6 ] [L0 ] [L1 ] [MD5:d9d1a722752fb7840601bc886a1a7078,109897675209fc9bd733a46689fc0053,6d48a58d0116f8da2b5de2483ce6492f]
    POC   14 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1901432 bits [Y 75.6635 dB    U 65.8817 dB    V 66.0672 dB] [ET     6 ] [L0 ] [L1 ] [MD5:05af3cede55291967d70046d413995ec,8fdfb6660eedbe18d6e50f1fbac67315,a658e6776632752d6f85e5e218208707]
    POC   15 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2297312 bits [Y 75.7841 dB    U 65.7038 dB    V 65.5377 dB] [ET     6 ] [L0 ] [L1 ] [MD5:1889353039a84d1ba44f7dfc77e16b58,92f20c8c61c399bd6469515b84805c4c,28c468b1748d33007ebdbc583705c555]
    POC   16 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2586448 bits [Y 78.5383 dB    U 65.4178 dB    V 65.4773 dB] [ET     6 ] [L0 ] [L1 ] [MD5:53c161080b58ad990a9e20a30715fbf5,b8f5596cd9ea9d9239408914a9c4776e,f746acd788c1d6fd53e6ab981a96044e]
    POC   17 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2209832 bits [Y 76.3319 dB    U 66.2039 dB    V 65.5768 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7d9dfb3856f465b26d4a7a2664cb5bf6,8fb01ff0cafc1ad8117eab02694087fd,78d2864c55102f03ee674cc7ff47df7c]
    POC   18 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2339336 bits [Y 76.9675 dB    U 65.1450 dB    V 64.9923 dB] [ET     6 ] [L0 ] [L1 ] [MD5:55f6c06c5342463e72fb57b526b70254,424c7023b73786a6c764e7136355e665,cf76e859c09bd7dad58abf6b04cedde6]
    POC   19 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1786064 bits [Y 77.3469 dB    U 65.4942 dB    V 65.5523 dB] [ET     5 ] [L0 ] [L1 ] [MD5:a7aa404117eb0ba9f0e97efaafcdb771,d270a0bf06e463d5f8eff1c20dd16f2b,0e5ded990e130b61f4123de6dec7bf5b]
    POC   20 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1886592 bits [Y 76.5336 dB    U 65.6585 dB    V 65.5743 dB] [ET     6 ] [L0 ] [L1 ] [MD5:59a6a02e506852dcc627c6c4cd2dcaa6,eae85d5639a53a407e439422a4fd98a0,47ee297b9e864b977b4d392660c08d39]
    POC   21 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2243224 bits [Y 78.0674 dB    U 65.8346 dB    V 65.2530 dB] [ET     6 ] [L0 ] [L1 ] [MD5:b7de4e013a0a99f201ceb5031bfcd978,febc88b91b799b81d18dc858d6592253,c9c82a7e45359a52e85ca3297b115b01]
    POC   22 TId: 0 ( I-SLICE, nQP 3 QP 3 )    1561944 bits [Y 75.4797 dB    U 67.0874 dB    V 66.1448 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ca087d1754342dfd188751493142dbca,777154ae2aec2959ceb77c75049fc3be,37ea53a81595093f27ed821dcb445cc9]
    POC   23 TId: 0 ( I-SLICE, nQP 3 QP 3 )    2635600 bits [Y 73.3253 dB    U 65.3381 dB    V 65.2827 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5ec1bb01d2f454069484a35e0cfb18f6,310123dbde916003b274b91169837e84,8905fffed189ba3121e9edf5b805205d]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  129380.2600   76.0014   65.3742   65.3157   69.3424  

# 2
    POC    0 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2224408 bits [Y 81.2896 dB    U 67.1681 dB    V 68.2383 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e083e073539b29da697f9817a922e2ef,571f3dff90b3896e6dff50f0e7eb1376,0d31bb1e88ae42e1a476544dbd2f3d00]
    POC    1 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2630976 bits [Y 77.5257 dB    U 67.8570 dB    V 67.4837 dB] [ET     6 ] [L0 ] [L1 ] [MD5:ec2de9ad5a3992101ce3227d371a48f1,99fc0a8ac5c10c7dc433f2e0cad14536,0a7b129e9723ece92c1f121222ec1ee2]
    POC    2 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2117936 bits [Y 81.5974 dB    U 68.2519 dB    V 69.1467 dB] [ET     6 ] [L0 ] [L1 ] [MD5:28c4161528521d4ae501e01d3f23efea,5a80dbe699146ee3cee1762f58e24bb7,f0910ef0c6d5224306921d339a81b347]
    POC    3 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2002680 bits [Y 83.1429 dB    U 70.1819 dB    V 70.3333 dB] [ET     6 ] [L0 ] [L1 ] [MD5:1ba56bc8a5a2751a8e99ced38d294a12,2167a924c18899bc1f59c45930d9b64c,6cec40e221dea4c0d0f2e91d5149dabe]
    POC    4 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2277616 bits [Y 79.9609 dB    U 67.9493 dB    V 67.6982 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e06a7c19285303b196152fcc082039d3,ca4267bd5f729556cd31018cd44505f5,54783abe6fc4a0013bc36402b8bd0339]
    POC    5 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1884144 bits [Y 82.8714 dB    U 67.9239 dB    V 67.9324 dB] [ET     5 ] [L0 ] [L1 ] [MD5:2edaf954d034362dfa3dbf81321031c9,c82f9c2478389ad31c51eed3845ef70e,9196929aab6cfcb4d8793334cba69dd4]
    POC    6 TId: 0 ( I-SLICE, nQP 2 QP 2 )    3048568 bits [Y 81.5974 dB    U 69.9274 dB    V 69.9542 dB] [ET     6 ] [L0 ] [L1 ] [MD5:ea852ed2e61744e448a3ac8b7c239214,689d046cbd626135550f88b58222a7d9,117790533c4bc8cd9f086f21b545cc05]
    POC    7 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1835072 bits [Y 79.8776 dB    U 69.5256 dB    V 69.0474 dB] [ET     6 ] [L0 ] [L1 ] [MD5:0d990a9bbd35e517109da3a89ee69235,2e724ad84987ea50cfb52503532cd356,6f11de897e5ac5bc7d8dfc3d9d4128f8]
    POC    8 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1707248 bits [Y 81.2668 dB    U 69.1467 dB    V 69.1300 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e987e3ea6a4304478c5d6d27fd61db9b,6920fcb40b03d05a6c0791bbef0aadfb,1d4f5114b922e910f1b71a2b5ae798e1]
    POC    9 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2045904 bits [Y 83.1781 dB    U 69.5439 dB    V 69.3175 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f5acbd785a6c5ac9bfcc3b11daa8e362,4694ba68e281c8cb7a407b29c7be881f,2c885275f6b5a347a022799368bfaab4]
    POC   10 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2030664 bits [Y 83.2493 dB    U 67.9493 dB    V 67.7344 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ea0f1319f6dfd36d1fb7402ef16ee72e,7fbabf14d67db25f7d343e1cc981d91d,b30deba273d1482eb6832320eba2bc73]
    POC   11 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2105544 bits [Y 84.2999 dB    U 68.9396 dB    V 70.1890 dB] [ET     5 ] [L0 ] [L1 ] [MD5:bb71050a52f61bf701675871030d0834,7653708fca053c421e178dc07a6b16c6,d398f52caf1bffcfdb5a656b8b5417f5]
    POC   12 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2509648 bits [Y 81.7726 dB    U 67.3451 dB    V 66.9709 dB] [ET     5 ] [L0 ] [L1 ] [MD5:dd411804658b00a78ee08f9b0c4cf827,aeba2b1cb9f1e7e6b9a466dea9b8fb5d,47c07b812fdb73621fb7938d8d053a57]
    POC   13 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2591904 bits [Y 80.8549 dB    U 67.5916 dB    V 67.4533 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7ae44cb17ef4417ddf073aefd00a108e,407d7fcb120169ce09c7eb3664ad8415,1920a1486fc0c542dfcf8c40026d452c]
    POC   14 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1948952 bits [Y 80.6926 dB    U 69.8809 dB    V 69.8348 dB] [ET     6 ] [L0 ] [L1 ] [MD5:6e1e55f1cfa4345e76ca06bda576e5bf,1bd29a175aef1c52dfd80e7948b915b1,de6b96cbdb5e95bf072237fd8837d496]
    POC   15 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2345584 bits [Y 82.2019 dB    U 68.7931 dB    V 68.8086 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5dd7c81e04de2815ed6cb81c68d55e36,29f61441521a4127ec70b92d3ece3674,3d745e18d629d9ba35201d63f937386e]
    POC   16 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2643632 bits [Y 83.2853 dB    U 69.7506 dB    V 69.4892 dB] [ET     6 ] [L0 ] [L1 ] [MD5:a02006d29acb96deec80434dee19a859,25d6ffda159982c2321895afdff3d9a5,77c934e591f8713c1d4096f0b56de8fd]
    POC   17 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2263072 bits [Y 81.3125 dB    U 70.2175 dB    V 69.3059 dB] [ET     6 ] [L0 ] [L1 ] [MD5:76be751c14e6e6c94d63fefb47549a60,9f9be3151b8ef02a2a2cda50f517e9d8,88a2ec429a882d8fd1a20f4c2ffb0713]
    POC   18 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2393264 bits [Y 82.6470 dB    U 69.0310 dB    V 68.6166 dB] [ET     6 ] [L0 ] [L1 ] [MD5:6bda4294f6b416a398dba5de70336fae,48310ef239e1629449c45d533ff1d090,1f816feada8e772238aa269506bfd322]
    POC   19 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1826144 bits [Y 82.9044 dB    U 70.2032 dB    V 69.8941 dB] [ET     6 ] [L0 ] [L1 ] [MD5:23493fbc753436af8c19dc3815ef8fb8,b193072137932c805bd68a76f212d44c,6b48b4566a9b71fb653190fa9cbfda53]
    POC   20 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1944024 bits [Y 82.1738 dB    U 70.1256 dB    V 69.7123 dB] [ET     6 ] [L0 ] [L1 ] [MD5:789feed27e1e13874e5aff11ed523cbe,6cb07b96b9fc00fce1a23c46f5e2dc52,bdc8a8be0028f0bddd08d439b6d3aec1]
    POC   21 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2297488 bits [Y 86.2956 dB    U 70.2750 dB    V 69.3583 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7554bece20c5a1cbe54229ce422c1c2e,a2b6446f2f5a9238c016b98f193541a1,c4940760a143c5932886bc75cd4fc565]
    POC   22 TId: 0 ( I-SLICE, nQP 2 QP 2 )    1599320 bits [Y 78.6613 dB    U 71.5148 dB    V 69.8153 dB] [ET     5 ] [L0 ] [L1 ] [MD5:88002fe4b69037fe67ce586505bbfae2,7807131a9a093b9d15408cbaa493631a,8954ee6b1ff53c23167926e342d1d645]
    POC   23 TId: 0 ( I-SLICE, nQP 2 QP 2 )    2688200 bits [Y 77.2018 dB    U 69.2598 dB    V 68.5431 dB] [ET     6 ] [L0 ] [L1 ] [MD5:d7b0e16365012a18b727f1eb153a04df,cd1b826ee759d4dafbfaad7e937f33ea,bcba59e907c8d4fe7ba197930f6b44e0]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  132404.9800   81.6609   69.0980   68.9170   73.1700 

# 1
    POC    0 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2290384 bits [Y 87.4495 dB    U 72.6283 dB    V 74.6720 dB] [ET     6 ] [L0 ] [L1 ] [MD5:c0d77f122ec0d8266c05495e9c3e2d94,67f5581e34176ccdb4c1ac789f0903c5,fc670f6b89a3cb6d8918b3655524c2b2]
    POC    1 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2702040 bits [Y 82.0359 dB    U 74.0082 dB    V 73.4929 dB] [ET     6 ] [L0 ] [L1 ] [MD5:27c04edcfcde51cad25fd1424d2b06dc,dbbd8e72ce5137d9f508879ca56b42cf,188612cac3c8360997ba07db909ce631]
    POC    2 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2183112 bits [Y 86.2238 dB    U 74.4016 dB    V 75.4797 dB] [ET     6 ] [L0 ] [L1 ] [MD5:72ffd55b8185806566da39154bf3094e,9a656e1c6dfe6880dccdfc9feeaabb63,210f66107c4348adcecc6daa3baf6d6a]
    POC    3 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2067528 bits [Y 87.7424 dB    U 76.7855 dB    V 75.7266 dB] [ET     6 ] [L0 ] [L1 ] [MD5:f1e8e23564747d2fed0b5a026b4adfd1,e2196058931cfa0f132bfb5b18b80be4,04491c13b18bf1eb2b9caa4ff91b52e0]
    POC    4 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2343736 bits [Y 87.4495 dB    U 74.2184 dB    V 74.6720 dB] [ET     6 ] [L0 ] [L1 ] [MD5:141de40e70466fa4f362469fae300f01,1469a804cc31b969ee51211956454b7a,5daf65e148e7ec6a4de83dec543e0627]
    POC    5 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1942576 bits [Y 89.3059 dB    U 74.2184 dB    V 74.3644 dB] [ET     6 ] [L0 ] [L1 ] [MD5:c2ca0b72b97bf00759f2fa5209ba7b0c,b8f8fb8267eb5bfcfcc06e2bcba25481,65dcfa2c4ed45629959f8350299a9559]
    POC    6 TId: 0 ( I-SLICE, nQP 1 QP 1 )    3135208 bits [Y 85.6261 dB    U 77.3377 dB    V 76.4729 dB] [ET     6 ] [L0 ] [L1 ] [MD5:12de09d0b18c651235399bbcf09bc7f2,ab37d3c3d6a11ae278bde6cd32915f73,1b3d0bd9a551c98dcd9b5c6e358150a7]
    POC    7 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1883880 bits [Y 85.6886 dB    U 76.4729 dB    V 75.2235 dB] [ET     5 ] [L0 ] [L1 ] [MD5:9b56dd8abe61c07f542f524fb147d156,201db2591321ffdb21a36ed2cf670ea3,4e25fce982810835675caf791e5cbb49]
    POC    8 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1757520 bits [Y 87.0014 dB    U 75.2690 dB    V 74.3644 dB] [ET     6 ] [L0 ] [L1 ] [MD5:5512c2764eb531618f0fbca5d134692e,4682ab477c6a5c270b9360a88d3a7809,0eea93ad0e829e053a9df0f0f767c423]
    POC    9 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2110392 bits [Y 90.6529 dB    U 75.9081 dB    V 75.0030 dB] [ET     6 ] [L0 ] [L1 ] [MD5:ecf7eca0ebe2b6f81f276a54b57b806c,d0250f9a373127b728135e9ad0dc42dd,8b53a9be7b0cfc488ab871627ee2bf2c]
    POC   10 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2090304 bits [Y 88.5141 dB    U 74.8343 dB    V 74.5540 dB] [ET     5 ] [L0 ] [L1 ] [MD5:bd765b633bb06375efe3e2311f6d4be7,c326e33a29ffa509baebdf8a06cf54ff,721e8ef40ff7d81bfbfb481c9b66bbb2]
    POC   11 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2168752 bits [Y 90.2750 dB    U 74.5154 dB    V 75.8034 dB] [ET     6 ] [L0 ] [L1 ] [MD5:7f08b7dfb8feece16aac3abc9182cc72,68571275c927f32aaaed875da72249c6,374a731e077f636b9251a739216d64ed]
    POC   12 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2584032 bits [Y 86.5952 dB    U 73.5081 dB    V 73.3436 dB] [ET     6 ] [L0 ] [L1 ] [MD5:6be5575b51a63354c19f4a065d3ebd38,9cc1c32ec64846b2ba42f7c3971b503a,674de510d10ecf4488a8604a7d6b00fc]
    POC   13 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2671624 bits [Y 86.0153 dB    U 73.6005 dB    V 72.9110 dB] [ET     6 ] [L0 ] [L1 ] [MD5:a23c380c4d255241f7f288c3ebf1894f,f89220004e79515b38a08d620de1632f,2336e5426ebe1bb70f28cf9fd81840d0]
    POC   14 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1998432 bits [Y 86.9171 dB    U 76.2956 dB    V 75.9883 dB] [ET     5 ] [L0 ] [L1 ] [MD5:ccce7bc3c637a0b1b93d883ee0b1a5b6,71fefa5319137b5156d97cbfe9d45440,15b452524cc50de1a81ade604215877c]
    POC   15 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2416568 bits [Y 87.1752 dB    U 75.1119 dB    V 74.8759 dB] [ET     6 ] [L0 ] [L1 ] [MD5:dfdc2073cc2584053a57b72cd664d848,685575911e2b6b3c16d974ce37dae3b8,6e421767524aeb1102f4e3ef8c8fdbc5]
    POC   16 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2720760 bits [Y 90.8549 dB    U 77.7223 dB    V 76.9171 dB] [ET     5 ] [L0 ] [L1 ] [MD5:e44cbec5a0fbfa37ec66d0591e842535,a8db63306b8365a01618c52aa043ab6b,a6882d940a02ddf2a5da2a51fff7423c]
    POC   17 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2332016 bits [Y 88.7623 dB    U 76.9844 dB    V 75.9081 dB] [ET     5 ] [L0 ] [L1 ] [MD5:4c46a91b3f0bb36b5e4029998f1c3c4a,ecbf6c0eaeec683f98de5aa2287fec7e,1e257767336e120e955b274684688a5f]
    POC   18 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2464024 bits [Y 89.7635 dB    U 76.7532 dB    V 76.6893 dB] [ET     5 ] [L0 ] [L1 ] [MD5:26f0193d4c73af6000b3ff053e8765b9,16e85b71d473e47d7f1700d24522e89e,fd6aa0ccc82c652680ff0ccd2a29c72b]
    POC   19 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1882944 bits [Y 86.9171 dB    U 76.0425 dB    V 77.4875 dB] [ET     5 ] [L0 ] [L1 ] [MD5:d4606a148d1718eae52dfefa5330c05a,555aef8b5532de79205ed0795aac0bc8,1ad9ba7d4828f155168fa8672e6fe67d]
    POC   20 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1999208 bits [Y 87.2647 dB    U 78.4661 dB    V 76.8508 dB] [ET     5 ] [L0 ] [L1 ] [MD5:f5a5cc09af7833588476292e70549431,0be01d25ee1c7bf2bf48355a78ef59fe,bd6efa06e8ab3f4c3ed6e595b12a9ed1]
    POC   21 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2363136 bits [Y 90.4598 dB    U 77.2647 dB    V 76.8180 dB] [ET     5 ] [L0 ] [L1 ] [MD5:7103a99819fdcab031c543687ded9e31,dd871d90be32511ad76ca6fd2b0c153d,8d60527b421bdf25d528f4f74b83c787]
    POC   22 TId: 0 ( I-SLICE, nQP 1 QP 1 )    1643544 bits [Y 86.9171 dB    U 76.3539 dB    V 75.4797 dB] [ET     4 ] [L0 ] [L1 ] [MD5:455b45489eef77cd143b2bd72f7bb0ad,49259f13d22cdeb3f3e1427a86cc0e49,0dded83f742bf7c1b1dc800df66d561a]
    POC   23 TId: 0 ( I-SLICE, nQP 1 QP 1 )    2764480 bits [Y 82.7417 dB    U 74.6323 dB    V 74.1649 dB] [ET     6 ] [L0 ] [L1 ] [MD5:499e1824c547f1a82c2d02fab09fc324,57cda22ff58aa9d055753dd2565fb04d,bccdc3c23e81fe0a7b05ca7c74540f4c]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  136290.5000   87.4312   75.5555   75.3026   79.4295  

# 0
    POC    0 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2365528 bits [Y 92.9377 dB    U 76.5336 dB    V 79.5439 dB] [ET     6 ] [L0 ] [L1 ] [MD5:ab6bf3203fc14358449e9a45755d305c,1f8a17a9999179115aaf9276bbea106c,801367f592b994054907926b44c794b7]
    POC    1 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2780968 bits [Y 86.0153 dB    U 80.3480 dB    V 79.3059 dB] [ET     6 ] [L0 ] [L1 ] [MD5:9b37e080c1c42738babe1717b5905ebc,2bcfe9849400e3415ae4b0df47f1472d,76098bc66c6ca4b8a630ffaacb149401]
    POC    2 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2251760 bits [Y 90.8549 dB    U 78.9717 dB    V 81.5244 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bc281fe98a12518576abaab6cf7ec787,bf7622e8818744ea199d88dc61b223b0,c957414a027e85f03b3bd542956fd578]
    POC    3 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2127168 bits [Y 92.0359 dB    U 84.0771 dB    V 82.4935 dB] [ET     6 ] [L0 ] [L1 ] [MD5:dcc30cccdd4c6b0414c7349a0c8e006c,d330cc58f258d4f7780e6175b4979284,c686688281d4cbab722aa8f10854f25b]
    POC    4 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2417304 bits [Y 91.0668 dB    U 80.8965 dB    V 81.9287 dB] [ET     5 ] [L0 ] [L1 ] [MD5:b3cfc49e049df2f9ffaf429f1ac15d6c,1d5f25dd40ae6473c9091de18d6af141,be26f8aae9750209e6e5f2effd21d0ff]
    POC    5 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2005480 bits [Y 91.7726 dB    U 79.7314 dB    V 79.1916 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bee0c8584c04a0ede504f2610297e37e,f6b495ed4ba8c9e53ace0d7fa485ca6b,9f66ca7d1dc6dcd42519ea4900b4c5cb]
    POC    6 TId: 0 ( I-SLICE, nQP 0 QP 0 )    3226000 bits [Y 92.0359 dB    U 86.5952 dB    V 86.2956 dB] [ET     6 ] [L0 ] [L1 ] [MD5:53682ae1460a4ca00036aab9a6ba297a,d5c1b6390df10d29fcac9f4d2d793c54,071495cde2c2122025ad18a188627e32]
    POC    7 TId: 0 ( I-SLICE, nQP 0 QP 0 )    1951544 bits [Y 92.0359 dB    U 80.6529 dB    V 79.2484 dB] [ET     5 ] [L0 ] [L1 ] [MD5:6f590f03173c5bb5d49579b6cd75d92d,5245724766e8674b3238c8d757bbe3a4,63f26fb5f0509fcf4e65b8f307f26b34]
    POC    8 TId: 0 ( I-SLICE, nQP 0 QP 0 )    1824600 bits [Y 88.7623 dB    U 78.5626 dB    V 76.8508 dB] [ET     5 ] [L0 ] [L1 ] [MD5:28f0ef0f0d1c300b8ac30c7e6ed27053,68d8afdc5cf0cc08c0fa9c2ed1dfc4db,9edf408109ea64bfaf2bd098dd75dc3e]
    POC    9 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2176800 bits [Y 92.6158 dB    U 83.7429 dB    V 81.3355 dB] [ET     6 ] [L0 ] [L1 ] [MD5:a7c571cc4492f7442f723e0f1026d703,5ca3d41479553b06057529de71b6dee2,cdb6975e080327b7fe0efa7974720ee6]
    POC   10 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2166008 bits [Y 98.0565 dB    U 81.1546 dB    V 79.1916 dB] [ET     5 ] [L0 ] [L1 ] [MD5:c706317f9b76b62daabcb78cd6d03f95,c8dca3aa6b53405dd5646579f65037ed,5f039512c947dade0b8703475d265c99]
    POC   11 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2233216 bits [Y 95.6261 dB    U 81.1546 dB    V 83.9068 dB] [ET     6 ] [L0 ] [L1 ] [MD5:71d2d202c62c2aaaf3ba7614339f11d4,73a9fdc2f316f97e1589b79632b9833d,2c6f5e97170987086e25b5c7a614c6e5]
    POC   12 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2664096 bits [Y 91.0668 dB    U 78.9717 dB    V 78.8657 dB] [ET     6 ] [L0 ] [L1 ] [MD5:9e2f8f30f1ef37757d2de0552429b681,59ef29a33bb88ee881a0448c3cf293db,bca3c3a1ea7060d70d44762720e36510]
    POC   13 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2755608 bits [Y 89.1635 dB    U 78.2338 dB    V 78.9717 dB] [ET     6 ] [L0 ] [L1 ] [MD5:78331a4daf0d584d3f7416bde646f8a9,b748a740133ac47afa363cc0958baf48,bcc544cc1132915e2c967e64c422cc47]
    POC   14 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2065144 bits [Y 92.6158 dB    U 81.8240 dB    V 82.0359 dB] [ET     6 ] [L0 ] [L1 ] [MD5:58a804166c0816a9273525de00d99127,6c2937ba46c841b71d5f835cbafa035e,c9504970af57165da4f46327fd693c6a]
    POC   15 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2490600 bits [Y 92.0359 dB    U 79.6680 dB    V 80.5746 dB] [ET     6 ] [L0 ] [L1 ] [MD5:773f6a0a109a8431cb58f49f913e7f98,fe87fe44fcdae9b44a38c2f29ad32bbb,a2a3b7fd609eeb4f9c4cfcc2cd661933]
    POC   16 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2800624 bits [Y 98.0565 dB    U 89.0256 dB    V 88.5141 dB] [ET     6 ] [L0 ] [L1 ] [MD5:be99360b6378e7bbfc7551db444cef64,6fe06e38c9160bb75ce26b5cba7160e0,a54ad09d992b4ff5663f7079d0a18bf8]
    POC   17 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2403200 bits [Y 95.6261 dB    U 85.0462 dB    V 81.8240 dB] [ET     6 ] [L0 ] [L1 ] [MD5:dbc769300252931e8073caa3b77bdf32,428eeb9f85974bc0256377edc962368b,596f26f268449aafc6bdbd673ddd92cf]
    POC   18 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2538968 bits [Y 96.2956 dB    U 85.7520 dB    V 83.2853 dB] [ET     7 ] [L0 ] [L1 ] [MD5:fcc95752cb3d728c4310e7174b19fe73,cafbedcce800e981cd39947795725331,f8715a8a54d41942fefce8cc09758ec9]
    POC   19 TId: 0 ( I-SLICE, nQP 0 QP 0 )    1947216 bits [Y 89.9274 dB    U 83.5849 dB    V 83.4325 dB] [ET     6 ] [L0 ] [L1 ] [MD5:e5cb2b9fb5b15380f38ed14ab27e93a1,c7acf4fa9cee0c4437c1da004906cb42,9689fc1c1b01304e32563a43237a4b18]
    POC   20 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2057248 bits [Y 92.6158 dB    U 84.6323 dB    V 83.7429 dB] [ET     6 ] [L0 ] [L1 ] [MD5:bd4141a5031347fd80c0c56133d952a3,deee63e5334424469dcbbd4acb584639,f21f5e1c50659edbe4fb8cdfa847b554]
    POC   21 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2433928 bits [Y 101.0668 dB    U 88.0565 dB    V 83.7429 dB] [ET     6 ] [L0 ] [L1 ] [MD5:fc552eb243c432d9ff14a15afd22dccc,2f8cdb3b2a10c247ece9e03172db50c5,7ce55d51b6c92322c77e3a7f0eb686d1]
    POC   22 TId: 0 ( I-SLICE, nQP 0 QP 0 )    1700544 bits [Y 91.5244 dB    U 81.2441 dB    V 79.0802 dB] [ET     5 ] [L0 ] [L1 ] [MD5:13fe6e62a21671c96855c1fb715c84a4,a92abe90525b01a926224e43af76e1f8,67ba67c1c124a0902a9cda5026e3912a]
    POC   23 TId: 0 ( I-SLICE, nQP 0 QP 0 )    2844928 bits [Y 87.4495 dB    U 81.0668 dB    V 80.2750 dB] [ET     6 ] [L0 ] [L1 ] [MD5:53a50f332863dc88e7cb50bd34036ecb,51e9d8ad0f6261d098ca8a1f3e408bda,6b84f2687839031041d734811515b8a5]
    SUMMARY --------------------------------------------------------
        Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR  
            24    a  140571.2000   92.5525   82.0636   81.4651   84.9744  

'''