"""Generate a textured synthetic room for reconstruction transport tests, in metres."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
from uuid import uuid4
import zipfile
import numpy as np
from PIL import Image, ImageDraw


def make_capture(output):
    rng = np.random.default_rng(28491)
    planes = []
    # Rectangle origin, unit U and V axes, dimensions. Add walls, floor and a raised box.
    rectangles = [([-2,0,-3],[1,0,0],[0,1,0],4,2.5),([-2,0,-3],[0,0,1],[0,1,0],5,2.5),
                  ([2,0,-3],[0,0,1],[0,1,0],5,2.5),([-2,0,-3],[1,0,0],[0,0,1],4,5),
                  ([-.45,0,-1.9],[1,0,0],[0,1,0],.9,.8),([-.45,.8,-1.9],[1,0,0],[0,0,-1],.9,.7),
                  ([-.45,0,-1.9],[0,0,-1],[0,1,0],.7,.8),([.45,0,-1.9],[0,0,-1],[0,1,0],.7,.8)]
    for number,(origin,u,v,width,height) in enumerate(rectangles):
        texture = Image.fromarray(rng.integers(65,210,(768,768,3),dtype=np.uint8)).resize((768,768))
        draw=ImageDraw.Draw(texture)
        for _ in range(400):
            x,y=rng.integers(0,740,2); size=int(rng.integers(5,25)); color=tuple(int(n) for n in rng.integers(0,255,3))
            draw.rectangle((x,y,x+size,y+size),fill=color)
        for row in range(8): draw.text((20,row*90+25),f'CALIBRATION SURFACE {number} ROW {row}',fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
        planes.append((np.array(origin,float),np.array(u,float),np.array(v,float),width,height,np.array(texture)))
    width,height=640,480
    yy,xx=np.mgrid[:height,:width]
    rays=np.stack([(xx-width/2)/520,-(yy-height/2)/520,-np.ones_like(xx)],axis=-1)
    frames=[]; photos={}
    for index,x in enumerate(np.linspace(-.65,.65,24)):
        position=np.array([x,1.3+.05*np.sin(index*.2),.7+.12*np.cos(index*.18)])
        forward=np.array([0,1.0,-2.5])-position;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        rotation=np.stack([right,up,-forward],axis=1)
        direction=rays@rotation.T
        nearest=np.full((height,width),np.inf); pixels=np.zeros((height,width,3),np.uint8)
        for origin,u,v,w,h,texture in planes:
            normal=np.cross(u,v);denominator=direction@normal
            distance=((origin-position)@normal)/np.where(abs(denominator)>1e-9,denominator,np.nan)
            points=position+direction*distance[...,None]-origin
            a=points@u;b=points@v
            valid=(distance>0)&(distance<nearest)&(a>=0)&(a<w)&(b>=0)&(b<h)
            tx=np.nan_to_num(a/w*767).astype(int).clip(0,767);ty=np.nan_to_num(b/h*767).astype(int).clip(0,767)
            pixels[valid]=texture[ty[valid],tx[valid]];nearest[valid]=distance[valid]
        photo=io.BytesIO();Image.fromarray(pixels).save(photo,'JPEG',quality=94)
        name=f'frame_{index+1:04d}.jpg';photos[name]=photo.getvalue()
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=position
        frames.append({'filename':name,'width':width,'height':height,'fx':520,'fy':520,'cx':width/2,'cy':height/2,
                       'cameraToWorld':pose.flatten(order='F').tolist(),'sha256':hashlib.sha256(photos[name]).hexdigest()})
    cloud=b'ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nproperty float confidence\nproperty uchar source\nend_header\n'
    manifest={'schemaVersion':1,'id':str(uuid4()),'images':frames,'product':'bridgecheck','scanName':'Synthetic calibration room',
              'createdAt':datetime.now(timezone.utc).isoformat(),'captureTimeZone':'UTC','captureUTCOffsetSeconds':0,
              'simulated':True,'coordinateSystem':'Synthetic metric GL camera poses; transport test only', 'warnings':['Synthetic scene. This does not verify physical phone accuracy.']}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('cloud.ply',cloud)
        for name,data in photos.items():archive.writestr(name,data)
    print(output)


if __name__=='__main__':
    import sys
    make_capture(Path(sys.argv[1]))
