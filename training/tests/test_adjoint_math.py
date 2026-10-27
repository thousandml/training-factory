"""Independent FP64 checks of FFT adjoints + physical column permutation.

These validate the mathematics without PyTorch/CUDA; they do not validate
quantization, mixed-precision rounding, native decoding or device dispatch.
"""
import unittest
import numpy as np


def swizzle(x):
    return x.reshape(-1,x.shape[-1]//1024,32,4,4,2).transpose(0,1,3,2,4,5).reshape(x.shape)


def unswizzle(x):
    return x.reshape(-1,x.shape[-1]//1024,4,32,4,2).transpose(0,1,3,2,4,5).reshape(x.shape)


def transform(x,inverse=False):
    z=x[...,0::2]+1j*x[...,1::2]
    y=(np.fft.ifft if inverse else np.fft.fft)(z,norm='ortho')
    return np.stack((y.real,y.imag),axis=-1).reshape(x.shape)


class AdjointMathTests(unittest.TestCase):
    def test_physical_scatter_index(self):
        # Matches inference.cu DECODE_AND_STORE_PASS's lane/chunk scatter.
        logical=np.arange(2048)
        physical=(logical//1024)*1024+((logical%32)//8)*256+((logical%1024)//32)*8+(logical%8)
        x=logical.reshape(1,-1)
        expected=np.empty_like(x)
        expected[0,physical]=logical
        np.testing.assert_array_equal(swizzle(x),expected)
        np.testing.assert_array_equal(unswizzle(expected),x)

    def test_fft_real_packed_adjoint(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(3,12));g=rng.normal(size=(3,12))
        np.testing.assert_allclose(np.sum(transform(x,True)*g),np.sum(x*transform(g)),rtol=1e-13)

    def test_full_vjp_and_directional_derivative(self):
        rng=np.random.default_rng(4)
        for rows,n,m in [(1,1024,12),(5,1024,16),(17,2048,24)]:
            with self.subTest(rows=rows,n=n,m=m):
                x=rng.normal(size=(rows,n));d=rng.normal(size=x.shape)
                g=rng.normal(size=(rows,m));w=rng.normal(size=(m,n))/np.sqrt(n)
                dl=rng.choice([-1.,1.],size=n);dr=rng.choice([-1.,1.],size=m)
                u,std=.13,.71
                def forward(v):
                    return transform(swizzle(transform(v*dl))@w.T,True)*dr*std+u*v.sum(-1,keepdims=True)
                gx=transform(unswizzle(transform(g*dr*std)@w),True)*dl+u*g.sum(-1,keepdims=True)
                np.testing.assert_allclose(np.sum(forward(x)*g),np.sum(x*gx),rtol=1e-11,atol=1e-11)
                eps=1e-5
                numerical=np.sum((forward(x+eps*d)-forward(x-eps*d))*g)/(2*eps)
                np.testing.assert_allclose(numerical,np.sum(gx*d),rtol=1e-8,atol=1e-8)

    def test_mean_term(self):
        g=np.arange(24,dtype=float).reshape(2,12)
        u=.125
        actual=np.broadcast_to(u*g.sum(-1,keepdims=True),(2,1024))
        self.assertTrue(np.all(actual[:,0]==actual[:,-1]))

if __name__=='__main__':
    unittest.main()
