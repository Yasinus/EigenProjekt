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