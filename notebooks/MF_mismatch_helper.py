# MF_mismatch_helper

import numpy as np
import sk_dsp_comm.digitalcom as dc
import scipy.signal as signal
import scipy.stats as stats

# Refactor some functions from scikit-dsp-comm

def cross_point(x,y,y_cross):
    """
    Find BEP curve crossing point using linear model
    Mark Wickert May 2018
    """
    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[0] - m*x[0]
    x_cross = (y_cross - b)/m
    return x_cross


def bit_errors(tx_data,rx_data,Ncorr = 1024,Ntransient = 0,disp_corr=False):
    """
    Count bit errors between a transmitted and received BPSK signal.
    Time delay between streams is detected as well as ambiquity resolution
    due to carrier phase lock offsets of :math:`k*\\pi`, k=0,1.
    The ndarray tx_data is Tx 0/1 bits as real numbers I.
    The ndarray rx_data is Rx 0/1 bits as real numbers I.
    Note: Ncorr needs to be even
    """
    
    # Remove Ntransient symbols and level shift to {-1,+1}
    tx_data = 2*tx_data[Ntransient:]-1
    rx_data = 2*rx_data[Ntransient:]-1
    # Correlate the first Ncorr symbols at four possible phase rotations
    R0 = np.fft.ifft(np.fft.fft(rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    R1 = np.fft.ifft(np.fft.fft(-1*rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    #Place the zero lag value in the center of the array
    R0 = np.fft.fftshift(R0)
    R1 = np.fft.fftshift(R1)
    R0max = np.max(R0.real)
    R1max = np.max(R1.real)
    R = np.array([R0max,R1max])
    Rmax = np.max(R)
    kphase_max = np.where(R == Rmax)[0]
    kmax = kphase_max[0]
    # Correlation lag value is zero at the center of the array
    if kmax == 0:
        lagmax = np.where(R0.real == Rmax)[0] - Ncorr/2
    elif kmax == 1:
        lagmax = np.where(R1.real == Rmax)[0] - Ncorr/2
    taumax = lagmax[0]
    if disp_corr == True:
        print('kmax =  %d, taumax = %d' % (kmax, taumax))

    # Count bit and symbol errors over the entire input ndarrays
    # Begin by making tx and rx length equal and apply phase rotation to rx
    if taumax < 0:
        tx_data = tx_data[int(-taumax):]
        tx_data = tx_data[:min(len(tx_data),len(rx_data))]
        rx_data = (-1)**kmax*rx_data[:len(tx_data)]
    else:
        rx_data = (-1)**kmax * rx_data[int(taumax):]
        rx_data = rx_data[:min(len(tx_data),len(rx_data))]
        tx_data = tx_data[:len(rx_data)]
    # Convert to 0's and 1's
    Bit_count = len(tx_data)
    tx_I = np.int16((tx_data.real + 1)/2)
    rx_I = np.int16((rx_data.real + 1)/2)
    Bit_errors = tx_I ^ rx_I
    return Bit_count,np.sum(Bit_errors)


def MPSK_BEP_thy(SNR_dB, M, EbN0_Mode = True):
    """
    Approximate the bit error probability of MPSK assuming Gray encoding
    
    Mark Wickert November 2018
    """
    if EbN0_Mode:
        EsN0_dB = SNR_dB + 10*np.log10(np.log2(M))
    else:
        EsN0_dB = SNR_dB
    Symb2Bits = np.log2(M)
    if M == 2:
        BEP = stats.norm.sf(np.sqrt(2*10**(EsN0_dB/10)))
    else:
        SEP = 2*stats.norm.sf(np.sqrt(2*10**(EsN0_dB/10))*np.sin(np.pi/M))
        BEP = SEP/Symb2Bits
    return BEP


def MPSK_BEP2EbN0(BEP, M):
    """
    Approximate the required EbN0 for given M and BEP
    
    Mark Wickert September 2019
    """
    # if EbN0_Mode:
    #     EsN0_dB = SNR_dB + 10*np.log10(np.log2(M))
    # else:
    #     EsN0_dB = SNR_dB
    Symb2Bits = np.log2(M)
    if M == 2:
        EsN0_dB = 10*np.log10(1/2*(stats.norm.isf(BEP))**2)
    else:
        EsN0_dB = 10*np.log10(1/2*(stats.norm.isf(BEP\
                  *Symb2Bits/2)/np.sin(np.pi/M))**2)
    return EsN0_dB - 10*np.log10(np.log2(M))


def QAM_BEP_thy(SNR_dB,M,EbN0_Mode = True):
    """
    Approximate the bit error probability of QAM assuming Gray encoding
    
    Mark Wickert November 2018
    """
    if EbN0_Mode:
        EsN0_dB = SNR_dB + 10*np.log10(np.log2(M))
    else:
        EsN0_dB = SNR_dB
    if M == 2:
        BEP = stats.norm.sf(np.sqrt(2*10**(EsN0_dB/10)))
    elif M > 2:
        SEP = 4*(1 - 1/np.sqrt(M))*stats.norm.sf(np.sqrt(3/(M-1)*10**(EsN0_dB/10)))
        BEP = SEP/np.log2(M)
    return BEP


def QAM_BEP2EbN0(BEP, M):
    """
    Approximate the required EbN0 for given M and BEP of 
    QAM assuming Gray encoding. Valid M values: 2, 4,
    16, 64, 256
    
    Mark Wickert September 2019
    """
    Symb2Bits = np.log2(M)
    if M == 2:
        EsN0_dB = 10*np.log10(1/2*(stats.norm.isf(BEP))**2)
    elif M > 2:
        EsN0_dB = 10*np.log10((M-1)/3*(stats.norm.isf(BEP*Symb2Bits/\
                              (4*(1 - 1/np.sqrt(M))))**2))        
    return EsN0_dB - 10*np.log10(np.log2(M))


def MPSK_gray_encode_bb(N_symb,Ns,M=4,pulse='rect',alpha=0.35,M_span=6,ext_data=None):
    """
    MPSK_gray_bb: A gray code mapped MPSK complex baseband transmitter 
    x,b,tx_data = MPSK_gray_bb(K,Ns,M)

    //////////// Inputs //////////////////////////////////////////////////
      N_symb = the number of symbols to process
          Ns = number of samples per symbol
           M = modulation order: 2, 4, 8, 16 MPSK
       alpha = squareroot raised cosine excess bandwidth factor.
               Can range over 0 < alpha < 1.
       pulse = 'rect', 'src', or 'rc'
    //////////// Outputs /////////////////////////////////////////////////
           x = complex baseband digital modulation
           b = transmitter shaping filter, rectangle or SRC
     tx_data = xI+1j*xQ = inphase symbol sequence + 
               1j*quadrature symbol sequence

    Mark Wickert November 2018
    """ 
    # Create a random bit stream then encode using gray code mapping
    # Gray code LUTs for 2, 4, 8, 16, and 32 MPSK
    # which employs M = 1, 2, 3, 4, and 5  bits per symbol  
    bin2gray1 = [0,1]
    bin2gray2 = [0,1,3,2]
    bin2gray3 = [0,1,3,2,7,6,4,5] 
    bin2gray4 = [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]
    bin2gray5 = [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10,31,30,
                 28,29,24,25,27,26,16,17,19,18,23,22,20,21]
    # Create the serial bit stream msb to lsb
    # except for the case M = 2
    N_word = int(np.log2(M))
    if N_symb == None:
        # Truncate so an integer number of symbols is formed
        N_symb = int(np.floor(len(ext_data)/N_word))
        data = ext_data[:N_symb*N_word]
    else:
        data = np.random.randint(0,2,size=int(np.log2(M))*N_symb)
    x_IQ = np.zeros(N_symb,dtype=np.complex128)
    # binary weights for converting binary to decimal using dot()
    bin_wgts = 2**np.arange(N_word-1,-1,-1)
    if M == 2: # Special case of BPSK for convenience
        x_IQ = 2*data - 1
    elif M == 4: # total constellation points
        for k in range(N_symb):
            word_phase = data[k*N_word:(k+1)*N_word]
            x_phase = 2*np.pi*bin2gray2[np.dot(word_phase,bin_wgts)]/M + np.pi/M
            x_IQ[k] = np.exp(1j*x_phase)
    elif M == 8:
        for k in range(N_symb):
            word_phase = data[k*N_word:(k+1)*N_word]
            x_phase = 2*np.pi*bin2gray3[np.dot(word_phase,bin_wgts)]/M
            x_IQ[k] = np.exp(1j*x_phase)
    elif M == 16:
        for k in range(N_symb):
            word_phase = data[k*N_word:(k+1)*N_word]
            x_phase = 2*np.pi*bin2gray4[np.dot(word_phase,bin_wgts)]/M
            x_IQ[k] = np.exp(1j*x_phase)
    elif M == 32:
        for k in range(N_symb):
            word_phase = data[k*N_word:(k+1)*N_word]
            x_phase = 2*np.pi*bin2gray5[np.dot(word_phase,bin_wgts)]/M
            x_IQ[k] = np.exp(1j*x_phase)
    else:
        raise ValueError('M must be 2, 4, 8, 16, or 32')        
    
    if Ns > 1:
        # Design the pulse shaping filter to be of duration 12 
        # symbols and fix the excess bandwidth factor at alpha = 0.35
        if pulse.lower() == 'src':
            b = dc.sqrt_rc_imp(Ns,alpha,M_span)
        elif pulse.lower() == 'rc':
            b = dc.rc_imp(Ns,alpha,M_span)    
        elif pulse.lower() == 'rect':
            b = np.ones(int(Ns)) #alt. rect. pulse shape
        else:
            raise ValueError('pulse shape must be src, rc, or rect')
        # Filter the impulse train signal
        x = signal.lfilter(b,1,dc.upsample(x_IQ,Ns))
        # Scale shaping filter to have unity DC gain
        b = b/sum(b)
        return x, b, data
    else:
        return x_IQ, 1, data


def QAM_gray_encode_bb(N_symb,Ns,M=4,pulse='rect',alpha=0.35,M_span=6,ext_data=None):
    """
    QAM_gray_bb: A gray code mapped QAM complex baseband transmitter 
    x,b,tx_data = QAM_gray_bb(K,Ns,M)
    
    Parameters
    ----------
    N_symb : The number of symbols to process
    Ns : Number of samples per symbol
    M : Modulation order: 2, 4, 16, 64, 256 QAM. Note 2 <=> BPSK, 4 <=> QPSK
    alpha : Square root raised cosine excess bandwidth factor.
            For DOCSIS alpha = 0.12 to 0.18. In general alpha can range over 0 < alpha < 1.
    pulse : 'rect', 'src', or 'rc'

    Returns
    -------
    x : Complex baseband digital modulation
    b : Transmitter shaping filter, rectangle or SRC
    tx_data : xI+1j*xQ = inphase symbol sequence + 1j*quadrature symbol sequence

    See Also
    --------
    QAM_gray_decode

    Examples
    --------
    
    
    
    """ 
    # Create a random bit stream then encode using gray code mapping
    # Gray code LUTs for 4, 16, 64, and 256 QAM
    # which employs M = 2, 4, 6, and 8 bits per symbol  
    bin2gray1 = [0,1]
    bin2gray2 = [0,1,3,2]
    bin2gray3 = [0,1,3,2,7,6,4,5] # arange(8) 
    bin2gray4 = [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]
    x_m = np.sqrt(M)-1
    # Create the serial bit stream [Ibits,Qbits,Ibits,Qbits,...], msb to lsb
    # except for the case M = 2
    if N_symb == None:
        # Truncate so an integer number of symbols is formed
        N_symb = int(np.floor(len(ext_data)/np.log2(M)))
        data = ext_data[:N_symb*int(np.log2(M))]
    else:
        data = np.random.randint(0,2,size=int(np.log2(M))*N_symb)
    x_IQ = np.zeros(N_symb,dtype=np.complex128)
    N_word = int(np.log2(M)/2)
    # binary weights for converting binary to decimal using dot()
    w = 2**np.arange(N_word-1,-1,-1)
    if M == 2: # Special case of BPSK for convenience
        x_IQ = 2*data - 1
        x_m = 1
    elif M == 4: # total constellation points
        for k in range(N_symb):
            wordI = data[2*k*N_word:(2*k+1)*N_word]
            wordQ = data[2*k*N_word+N_word:(2*k+1)*N_word+N_word]
            x_IQ[k] = (2*bin2gray1[np.dot(wordI,w)] - x_m) + \
                   1j*(2*bin2gray1[np.dot(wordQ,w)] - x_m)
    elif M == 16:
        for k in range(N_symb):
            wordI = data[2*k*N_word:(2*k+1)*N_word]
            wordQ = data[2*k*N_word+N_word:(2*k+1)*N_word+N_word]
            x_IQ[k] = (2*bin2gray2[np.dot(wordI,w)] - x_m) + \
                   1j*(2*bin2gray2[np.dot(wordQ,w)] - x_m)
    elif M == 64:
        for k in range(N_symb):
            wordI = data[2*k*N_word:(2*k+1)*N_word]
            wordQ = data[2*k*N_word+N_word:(2*k+1)*N_word+N_word]
            x_IQ[k] = (2*bin2gray3[np.dot(wordI,w)] - x_m) + \
                   1j*(2*bin2gray3[np.dot(wordQ,w)] - x_m)
    elif M == 256:
        for k in range(N_symb):
            wordI = data[2*k*N_word:(2*k+1)*N_word]
            wordQ = data[2*k*N_word+N_word:(2*k+1)*N_word+N_word]
            x_IQ[k] = (2*bin2gray4[np.dot(wordI,w)] - x_m) + \
                   1j*(2*bin2gray4[np.dot(wordQ,w)] - x_m)
    else:
        raise ValueError('M must be 2, 4, 16, 64, 256')        
    
    if Ns > 1:
        # Design the pulse shaping filter to be of duration 12 
        # symbols and fix the excess bandwidth factor at alpha = 0.35
        if pulse.lower() == 'src':
            b = dc.sqrt_rc_imp(Ns,alpha,M_span)
        elif pulse.lower() == 'rc':
            b = dc.rc_imp(Ns,alpha,M_span)    
        elif pulse.lower() == 'rect':
            b = np.ones(int(Ns)) #alt. rect. pulse shape
        else:
            raise ValueError('pulse shape must be src, rc, or rect')
        # Filter the impulse train signal
        x = signal.lfilter(b,1,dc.upsample(x_IQ,Ns))
        # Scale shaping filter to have unity DC gain
        b = b/sum(b)
        return x/x_m, b, data
    else:
        return x_IQ/x_m, 1, data