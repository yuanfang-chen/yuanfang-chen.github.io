## Chapter 5 Exercise

1. No. Because each thread uses it own data; there are data sharing across threads
2. "I did some mental calculation"
3. RAW, WAR hazards can happen. make the results wrong. 
4. shared memory let threads share data; registers are private to each thread.
5. 1/32
6. 512*1000=512000
7. 1000
8. a. N b. N/T
9. 36 FLOP / (7*4B) = 1.29 FLOP/B; 
    a. 200/100 = 2 FLOP/B; memory bound
    b. 300/250 = 1.2 FLOP/a; compute bound
10. 
    a. `BLOCK_SIZE` is typo, it should `BLOCK_WIDTH`. If `BLOCK_WIDTH` is 1, the code is correct
    b. add `__syncthreads()` after line 10
11. 
    a. reg. 1024
    b. local mem. 1024
    c. shared mem. 8
    d. shared mem. 8
    e. 129*4 = 516
    f. a is accecced 4 times at line 7. 10 FLOP at line 14. b is accessed 1 time at line 14. 10/((4+1)*4) = 0.5 FLOP/B
12.
    a. 2048/64=32 block. 2048*27=55296 regs. 4KB*32=128KB shared memory. NOT full occupancy 
    b. 2048/256=8 block. 2048*31=63488 regs. 8KB*8=64KB shared memory. full occupancy


