import glob, io, os, pickle, sys
from multiprocessing import Pool
from PIL import Image, ImageOps

SIZE = 32

def sig_from_image(im, size=SIZE):
    im = ImageOps.exif_transpose(im)
    im = im.convert('L').resize((size, size), Image.BILINEAR)
    return im.tobytes()

def raw_sig(path, size=SIZE):
    try:
        if path.lower().endswith('.cr2'):
            b = open(path, 'rb').read(8_000_000)
            best = None
            i = 0
            while True:
                i = b.find(b'\xff\xd8\xff', i)
                if i < 0:
                    break
                try:
                    im = Image.open(io.BytesIO(b[i:]))
                    if im.size[0] > 1000:
                        im.draft('L', (400, 400))
                        im.load()
                        best = im
                        break
                except Exception:
                    pass
                i += 2
            if best is None:
                return path, None
            return path, sig_from_image(best, size)
        im = Image.open(path)
        im.draft('L', (400, 400))
        im.load()
        return path, sig_from_image(im, size)
    except Exception as e:
        return path, None

def ds_sig(path, size=SIZE):
    try:
        im = Image.open(path)
        im.load()
        return path, sig_from_image(im, size)
    except Exception:
        return path, None

if __name__ == '__main__':
    raws = [f for f in glob.glob('*_raw/**/*.*', recursive=True)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.cr2')]
    dss = glob.glob('*_datasets/**/*.jpg', recursive=True)
    print(len(raws), len(dss), flush=True)
    with Pool(8) as p:
        r = dict(p.imap_unordered(raw_sig, raws, chunksize=8))
        print('raw done', sum(v is None for v in r.values()), 'failed', flush=True)
        d = dict(p.imap_unordered(ds_sig, dss, chunksize=16))
        print('ds done', sum(v is None for v in d.values()), 'failed', flush=True)
    pickle.dump({'raw': r, 'ds': d}, open('_features.pkl', 'wb'))
    print('saved', flush=True)
