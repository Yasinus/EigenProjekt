import numpy as np
class Oklab:

    def __init__(self, bgr):
        self.rgb = bgr[:, :, ::-1]
        self.xyz = self.rgb2xyz()
        self.lms = self.xyz2lms()
        self.oklab = self.lms2oklab()
        self.lCh = self.oklab2lCh()



    def rgb2xyz(self):
        rgb = self.rgb / 255.0
        rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        rgb = rgb * 100.0
        xyz = np.dot(rgb, np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                    [0.2119034982, 0.6806995451, 0.1073969566],
                                    [0.0883024619, 0.2817188376, 0.6299787005]]))
        return xyz



    def xyz2lms(self):
        lms = np.dot(self.xyz, np.array([[0.4002, 0.7076, -0.0808],
                                        [-0.2263, 1.1653, 0.0457],
                                        [0, 0, 0.9182]]))
        lms = np.where(lms <= 0, 1e-10, lms)
        return lms



    def lms2oklab(self):
        lms = self.lms
        lms = np.where(lms <= 0, 1e-10, lms)
        lms = np.log10(lms)
        oklab = np.dot(lms, np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                                    [1.9779984951, -2.4285922050, 0.4505937099],
                                    [0.0259040371, 0.7827717662, -0.8086757660]]))
                                    
        return oklab



    def oklab2lCh(self):
        oklab = self.oklab
        l = oklab[:,:,0]
        a = oklab[:,:,1]
        b = oklab[:,:,2]
        c = np.sqrt(a ** 2 + b ** 2)
        h = np.arctan2(b, a)
        return (np.array([l, c, h]) * 255).astype(np.uint8)



    def get_oklab(self):
        return self.oklab
    

    
    def get_lCh(self):
        return self.lCh
    
class reverse_Oklab:
    def __init__(self, lCh):
        self.lCh = lCh
        self.oklab = self.lCh2oklab()
        self.lms = self.oklab2lms()
        self.xyz = self.lms2xyz()
        self.rgb = self.xyz2rgb()
        self.bgr = self.rgb[:, :, ::-1]
    
    def lCh2oklab(self):
        l = self.lCh[0] / 255.0
        c = self.lCh[1] / 255.0
        h = self.lCh[2] / 255.0
        a = c * np.cos(h)
        b = c * np.sin(h)
        oklab = np.dstack((l, a, b))
        return oklab
    
    def oklab2lms(self):
        oklab = self.oklab
        l = oklab[:,:,0]
        a = oklab[:,:,1]
        b = oklab[:,:,2]
        lms = np.dot(oklab, np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                                    [1.9779984951, -2.4285922050, 0.4505937099],
                                    [0.0259040371, 0.7827717662, -0.8086757660]]))
        lms = 10 ** lms
        return
    
    def lms2xyz(self):
        lms = self.lms
        l = lms[:,:,0]
        m = lms[:,:,1]
        s = lms[:,:,2]
        xyz = np.dot(lms, np.array([[1.862067855087233, -1.011254630531512, 0.149186575585921],
                                    [0.387526968021060, 0.621447441818524, -0.008973985785558],
                                    [-0.015841813798413, -0.034610936001449, 1.049404351494094]]))
        return xyz
    
    def xyz2rgb(self):
        xyz = self.xyz / 100.0
        r = np.dot(xyz, np.array([[3.2404542, -1.5371385, -0.4985314],
                                    [-0.9692660, 1.8760108, 0.0415560],
                                    [0.0556434, -0.2040259, 1.0572252]]))
        r = np.where(r > 0.0031308, 1.055 * (r ** (1 / 2.4)) - 0.055, 12.92 * r)
        r = np.clip(r, 0, 1)
        return r * 255.0
    
    def get_bgr(self):
        return self.bgr
