## Chapter 4 Exercise

1. 

a. 4 warps per block
b. 32 warps per grid (8 block per grid)
c. 
      i. for 40, [0,32), [32,64), two warps are active; for >=104, [96, 128), one warp is active; total 3 warps are active in a block; 3*8=24 warps are active in a grid
     ii. 2 warps are divergent
    iii. 100%
     iv. 8/32 = 25%
      v. (128-104)/32 = 75%
d.
      i. 32
     ii. 32
    iii. 50%
e.
      i. 3
     ii. 2

2. 2048
3. 1
4. max = 3.0; (1.0 + 0.7 + 0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1) / (8*3.0) = 17.08%
5. bad idea. threads of a single warp could still out of sync.
6. b
7. 
    a. 50%
    b. 50%
    c. 50%
    d. 100%
    e. 100%

8. 
    a. can get full occupancy. 16 blocks; 2048*30 = 61440 regs
    b. cannot get full occupancy. need 64 blocks but limited by 32 blocks per SM
    c. cannot get full occupancy. need 2048*34=69632 regs but limited by 65536 regs per SM

9.
$$32 \times 32 = 1024 \text{ threads per block}$$ but only 512 threads per block is allowed
