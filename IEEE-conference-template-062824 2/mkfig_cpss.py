import math, os, matplotlib
from PIL import ImageFont
TTF={s:os.path.join(os.path.dirname(matplotlib.__file__),'mpl-data/fonts/ttf/DejaVuSans%s.ttf'%s)
     for s in ('','-Bold','-Oblique')}
_c={}
def wd(t,size,style=''):
    k=(size,style)
    if k not in _c: _c[k]=ImageFont.truetype(TTF[style],int(round(size*64)))
    return _c[k].getlength(t)/64.0
def rwd(parts,size,style=''):
    return sum(wd(t,size if k=='base' else size*0.66,style) for t,k in parts)

W,H=950,448
o=[];add=o.append
F='DejaVu Sans, Helvetica, Arial, sans-serif'
INK='#1f2937';BLUE='#1d4ed8';GRN='#15803d';GRNF='#dcfce7';RED='#b91c1c';REDF='#fee2e2'
PUR='#6d28d9';PURF='#f5f3ff';GREY='#9ca3af';GREYF='#e5e7eb';SL='#475569'
FBOX,FLAB=26,23

def T(x,y,s,size,fill=INK,anchor='middle',style=''):
    st='italic' if style=='-Oblique' else 'normal'
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="{F}" font-style="{st}" font-weight="{"bold" if style=="-Bold" else "normal"}">{s}</text>')
def run(x,y,parts,size,fill=INK,anchor='middle',style=''):
    tot=rwd(parts,size,style); cur=x-tot/2 if anchor=='middle' else (x-tot if anchor=='end' else x)
    for t,k in parts:
        sz=size if k=='base' else size*0.66
        dy=0 if k=='base' else (size*0.25 if k=='sub' else -size*0.40)
        T(cur,y+dy,t,round(sz,1),fill,'start',style); cur+=wd(t,sz,style)
    return tot
def CT(x,y,lab,size,fill=SL,style='-Oblique'):
    half=wd(lab,size,style)/2
    add_x=min(max(x,half+6),W-half-6)
    T(add_x,y,lab,size,fill,'middle',style)
def R(x,y,w,h,fill='#fff',stroke=INK,sw=2.2,rx=9,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def AR(x1,y1,x2,y2,c=INK,sw=2.4,hd=13):
    a=math.atan2(y2-y1,x2-x1);bx,by=x2-hd*math.cos(a),y2-hd*math.sin(a)
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{c}" stroke-width="{sw}" stroke-linecap="round"/>')
    p=[(x2,y2),(bx-hd*.48*math.sin(a),by+hd*.48*math.cos(a)),(bx+hd*.48*math.sin(a),by-hd*.48*math.cos(a))]
    add('<polygon points="'+' '.join(f'{u:.1f},{v:.1f}' for u,v in p)+f'" fill="{c}"/>')
def hat(cx,y,w=11):
    add(f'<path d="M{cx-w/2:.1f},{y} L{cx:.1f},{y-7} L{cx+w/2:.1f},{y}" fill="none" stroke="{INK}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>')

def lay(items,total,gap,y):
    """items: list of (width, draw_fn). Centres the row; returns list of (x,width)."""
    span=sum(w for w,_ in items)+gap*(len(items)-1)
    x=(total-span)/2; out=[]
    for i,(w,fn) in enumerate(items):
        fn(x,y,w); out.append((x,w))
        if i<len(items)-1: AR(x+w+5,y+ITEM_H/2,x+w+gap-5,y+ITEM_H/2)
        x+=w+gap
    return out

ITEM_H=68
def vec(cells=3):
    w=cells*22
    def fn(x,y,_w):
        R(x,y+14,w,40,'#fff',BLUE,2.6,0)
        for k in range(1,cells): add(f'<line x1="{x+k*22}" y1="{y+14}" x2="{x+k*22}" y2="{y+54}" stroke="#bfdbfe" stroke-width="1.5"/>')
        for k in (0,2): add(f'<rect x="{x+k*22+5}" y="{y+25}" width="12" height="17" rx="2" fill="#111827"/>')
    return w,fn
def poolbox():
    w=max(wd('TM clause',FBOX),wd('pool',FBOX))+40
    def fn(x,y,_w):
        R(x,y,_w,ITEM_H,'#eef2ff',INK,2.6,10)
        T(x+_w/2,y+28,'TM clause',FBOX); T(x+_w/2,y+54,'pool',FBOX)
    return w,fn

# ---------------- ROW A ----------------
yA=16
def zmat(x,y,w):
    R(x,y+2,w,64,'#fff',BLUE,2.8,0)
    for g in (1,2,3): add(f'<line x1="{x+g*w/4:.1f}" y1="{y+2}" x2="{x+g*w/4:.1f}" y2="{y+66}" stroke="#dbeafe" stroke-width="1.5"/>')
    for g in (1,2): add(f'<line x1="{x}" y1="{y+2+g*21.3:.1f}" x2="{x+w}" y2="{y+2+g*21.3:.1f}" stroke="#dbeafe" stroke-width="1.5"/>')
    for a,b in [(.06,.06),(.55,.05),(.28,.36),(.72,.38),(.12,.70),(.60,.72)]:
        add(f'<rect x="{x+a*w:.1f}" y="{y+6+b*54:.1f}" width="18" height="9" rx="1.5" fill="#111827"/>')
rowA=lay([vec(3),poolbox(),(126,zmat)],W,34,yA)
for (x,w),lab in zip(rowA,['validation half',None,'Z : clause activations']):
    if lab: CT(x+w/2,yA+96,lab,FLAB)
AR(rowA[2][0]+rowA[2][1]/2,yA+106,rowA[2][0]+rowA[2][1]/2,132)

# ---------------- ROW B : CPSS band ----------------
yB=138; BH=176
R(12,yB,W-24,BH,'#f8fafc','#94a3b8',2,13,'8 6')
T(28,yB+26,'one-vs-rest, repeated per family  ℓ = 1…6',FLAB,SL,'start','-Oblique')
hy=yB+40
hw=max(wd('B complementary pairs',FLAB),wd('→ 2B half-fits',FLAB),
       rwd([('ℓ','base'),('1','sub'),('\u00a0logistic · κ grid (6 pts)','base')],FLAB))+34
chw=6*26+wd('π',FLAB)+34
sw_=max(rwd([('S','base'),('+','sup'),('ℓ','sub'),('\u00a0\u00a0inculpatory','base')],FBOX),
        rwd([('S','base'),('−','sup'),('ℓ','sub'),('\u00a0\u00a0exculpatory','base')],FBOX))+40
gap=30; span=hw+chw+sw_+2*gap; x0=(W-span)/2
R(x0,hy,hw,104,PURF,PUR,2.4,10)
T(x0+hw/2,hy+30,'B complementary pairs',FLAB,PUR)
T(x0+hw/2,hy+59,'→ 2B half-fits',FLAB,PUR)
run(x0+hw/2,hy+88,[('ℓ','base'),('1','sub'),('\u00a0logistic · κ grid (6 pts)','base')],FLAB,PUR)
cx=x0+hw+gap
AR(x0+hw+4,hy+52,cx-4,hy+52)
base=hy+86
for i,(top,col) in enumerate([(hy+10,GRN),(hy+44,GREY),(hy+20,GRN),(hy+56,GREY),(hy+4,GRN),(hy+62,GREY)]):
    add(f'<rect x="{cx+8+i*26}" y="{top}" width="20" height="{base-top}" fill="{GRNF if col==GRN else GREYF}" stroke="{col}" stroke-width="2"/>')
add(f'<line x1="{cx+2}" y1="{base}" x2="{cx+162}" y2="{base}" stroke="{INK}" stroke-width="2.4"/>')
add(f'<line x1="{cx}" y1="{hy+36}" x2="{cx+168}" y2="{hy+36}" stroke="{RED}" stroke-width="2.6" stroke-dasharray="10 7"/>')
run(cx+172,hy+41,[('π','base'),('thr','sub')],FLAB,RED,'start')
tt=run(cx+82,base+34,[('stability votes\u00a0π','base'),('j','sub')],FLAB,SL,'middle','-Oblique')
hat(cx+82-tt/2+wd('stability votes\u00a0π',FLAB,'-Oblique')-wd('π',FLAB,'-Oblique')/2, base+20)
sx=cx+chw+gap
R(sx,hy,sw_,46,GRNF,GRN,2.6,10);  run(sx+sw_/2,hy+31,[('S','base'),('+','sup'),('ℓ','sub'),('\u00a0\u00a0inculpatory','base')],FBOX,GRN)
R(sx,hy+58,sw_,46,REDF,RED,2.6,10); run(sx+sw_/2,hy+89,[('S','base'),('−','sup'),('ℓ','sub'),('\u00a0\u00a0exculpatory','base')],FBOX,RED)
AR(cx+chw-18,hy+30,sx-6,hy+23,GRN)
AR(cx+chw-18,hy+66,sx-6,hy+81,RED)

# ---------------- ROW C : scoring ----------------
yC=yB+BH+22
def score(x,y,w):
    R(x,y,w,ITEM_H,'#fff',INK,2.6,10)
    add(f'<rect x="{x+16}" y="{y+17}" width="76" height="34" rx="4" fill="{GRN}"/>'); T(x+54,y+41,'Σ⁺',24,'#fff','middle','-Bold')
    T(x+112,y+43,'−',28,INK,'middle','-Bold')
    add(f'<rect x="{x+132}" y="{y+17}" width="56" height="34" rx="4" fill="{RED}"/>'); T(x+160,y+41,'Σ⁻',24,'#fff','middle','-Bold')
    run(x+w-16,y+42,[('s','base'),('ℓ','sub'),('(x)','base')],FBOX,INK,'end')
def argm(x,y,w):
    R(x,y,w,ITEM_H,'#f1f5f9',INK,2.6,10)
    t=run(x+w/2,y+44,[('y = arg max','base'),('ℓ','sub')],FBOX)
    hat(x+w/2-t/2+wd('y',FBOX)/2, y+22)
scw=16+76+36+56+20+rwd([('s','base'),('ℓ','sub'),('(x)','base')],FBOX)+16
amw=rwd([('y = arg max','base'),('ℓ','sub')],FBOX)+40
rowC=lay([vec(3),poolbox(),vec(3),(scw,score),(amw,argm)],W,30,yC)
for (x,w),lab in zip(rowC,['held-out flow  x',None,'z(x)','match score, Eq. (2)','predicted family']):
    if lab: CT(x+w/2,yC+96,lab,FLAB)

svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
     f'<rect width="{W}" height="{H}" fill="#fff"/>'+''.join(o)+'</svg>')
open(os.path.join(os.path.dirname(__file__),'fig_cpss_new.svg'),'w').write(svg)
print('ok',W,H,'aspect',round(W/H,2),'rowC span',rowC[-1][0]+rowC[-1][1])
