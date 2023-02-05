# Pseudo code for LSTM computation

sequence_len = 10 

for i in range (0,sequence_len):
    
    # if we are on the initial step
    # initialize h(t-1) and c(t-1) 
    # randomly
    if i==0:
        ht_1 = random ()
        ct_1 = random ()
       
    else: 
        ht_1 = h_t
        ct_1 = c_t
    
    f_t = sigmoid ( 
                     matrix_mul(Wf, xt) +
                     matrix_mul(Uf, ht_1) +
                     bf
                   )
    i_t = sigmoid ( 
                     matrix_mul(Wi, xt) +
                     matrix_mul(Ui, ht_1) +
                     bi
                   )
    o_t = sigmoid ( 
                     matrix_mul(Wo, xt) +
                     matrix_mul(Uo, ht_1) +
                     bo
                   )
    cp_t = tanh   ( 
                     matrix_mul(Wc, xt) +
                     matrix_mul(Uc, ht_1) +
                     bc
                   )

    c_t  = element_wise_mul(f_t, ct_1) + 
           element_wise_mul(i_t, cp_t) 
    h_t  = element_wise_mul(o_t, tanh(c_t))