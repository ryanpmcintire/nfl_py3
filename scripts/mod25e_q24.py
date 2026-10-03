import sys,glob
sys.path.insert(0,'scripts')
import numpy as np,pandas as pd
import mod25e_f2pr as m

rng=np.random.default_rng(0)
B=400

def load():
    R=pd.read_parquet(m.OUT/'real_fit.parquet')
    R=R[(R.season>=2009)&(R.season<=2017)].copy()
    R['gid']=R.game_id.astype(str)
    fr=[]
    for sd in (11,12,13):
        for f in sorted(glob.glob(f'artifacts/mod25e3/e5_crHpq_s{sd}/play_*_*.parquet')):
            w,s=(int(x) for x in f.split('play_')[1][:-8].split('_'))
            if s<2: continue
            d=pd.read_parquet(f);d['gid']=f'{sd}_{w}_{s}_'+d.g.astype(int).astype(str);fr.append(d)
    S=pd.concat(fr)
    for D in (R,S):
        D['pts']=D.po+D.pdf
        D['tdp']=(D.po>=6)&D.code.isin([0,1])
    return R,S

def boot(a,b,na,nb):
    d=[]
    for _ in range(B):
        ia=rng.integers(0,len(a),len(a));ib=rng.integers(0,len(b),len(b))
        d.append(b[ib].sum()/len(b)-a[ia].sum()/len(a))
    d=np.array(d)
    return d.mean(),np.percentile(d,2.5),np.percentile(d,97.5),(d>0).mean()

def pergame(D,mask,col,gids):
    s=D[mask].groupby('gid')[col].sum()
    return s.reindex(gids).fillna(0).values

def seg(R,S):
    gr=R.gid.unique();gs=S.gid.unique()
    print('=== part 1: points per game sim minus real by window and score state (sd=posteam lead)')
    def tl(D):
        return np.select([D.sd<0,D.sd==0],['trail','tied'],'lead')
    R['st']=tl(R);S['st']=tl(S)
    for D in (R,S):
        D['rem']=np.where(D.qtr==2,D.gsr-1800,np.where(D.qtr==4,D.gsr,np.nan))
        D['win']=np.where(D.rem<=120,'last2',np.where(D.rem<=300,'2to5','rest'))
    for q in (2,4):
        for w in ('last2','2to5','rest'):
            for st in (None,'trail','tied','lead'):
                mr=(R.qtr==q)&(R.win==w);ms=(S.qtr==q)&(S.win==w)
                if st: mr&=R.st==st;ms&=S.st==st
                a=pergame(R,mr,'pts',gr);b=pergame(S,ms,'pts',gs)
                e=boot(a,b,len(gr),len(gs))
                print(f'q{q} {w:5s} {st or "all":5s} real {a.mean():.3f} sim {b.mean():.3f} diff {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] P+ {e[3]:.3f}')
    for q in (2,4):
        for w in ('last2','2to5','rest'):
            mr=(R.qtr==q)&(R.win==w);ms=(S.qtr==q)&(S.win==w)
            print(f'q{q} {w} snaps/g real {mr.sum()/len(gr):.3f} sim {ms.sum()/len(gs):.3f} pts/snap real {R[mr].pts.sum()/mr.sum():.4f} sim {S[ms].pts.sum()/ms.sum():.4f}')

def goal(R,S):
    gr=R.gid.unique();gs=S.gid.unique()
    print('=== part 2: run/pass near goal; snaps per game, pass share, TD/play by type, sim minus real TD pts/g')
    bands=[('yl1-3',1,3),('yl4-5',4,5),('yl6-10',6,10),('yl11-20',11,20),('yl21-40',21,40),('yl41-60',41,60),('yl61+',61,100)]
    for gtg in (False,True):
        for nm,lo,hi in bands:
            if gtg and hi>20: continue
            rows={}
            for k,D,g in (('R',R,gr),('S',S,gs)):
                mk=D.code.isin([0,1])&(D.yl>=lo)&(D.yl<=hi)&(D.down<=4)
                if gtg: mk&=(D.dist>=D.yl)
                X=D[mk]
                ps=(X.code==1)
                rows[k]=dict(n=len(X)/len(g),ps=ps.mean(),tdp=X[ps].tdp.mean(),tdr=X[~ps].tdp.mean(),pt=X[ps].tdp.sum()/len(g),rt=X[~ps].tdp.sum()/len(g),yp=X[ps].yards.mean(),yr=X[~ps].yards.mean())
            r,s=rows['R'],rows['S']
            print(f'{"gtg " if gtg else "all "}{nm:8s} snaps/g {r["n"]:.3f}/{s["n"]:.3f} passShare {r["ps"]:.3f}/{s["ps"]:.3f} TD|pass {r["tdp"]:.4f}/{s["tdp"]:.4f} TD|run {r["tdr"]:.4f}/{s["tdr"]:.4f} passTD/g {r["pt"]:.4f}/{s["pt"]:.4f} runTD/g {r["rt"]:.4f}/{s["rt"]:.4f} ypp pass {r["yp"]:.2f}/{s["yp"]:.2f} run {r["yr"]:.2f}/{s["yr"]:.2f}')
    print('--- bootstrap TD/g sim-real (pass, run) by yl band')
    for nm,lo,hi in bands:
        out=[]
        for c in (1,0):
            a=pergame(R,R.code.isin([c])&R.tdp&(R.yl>=lo)&(R.yl<=hi),'tdp',gr);b=pergame(S,S.code.isin([c])&S.tdp&(S.yl>=lo)&(S.yl<=hi),'tdp',gs)
            e=boot(a,b,len(gr),len(gs));out.append(f'{"pass" if c else "run"} {e[0]:+.4f} [{e[1]:+.4f},{e[2]:+.4f}] P+ {e[3]:.2f}')
        print(nm,' | '.join(out))
    print('--- long pass TDs (yards>=40 scoring plays) and 40+ yard pass plays')
    for k,D,g in (('R',R,gr),('S',S,gs)):
        p=D[D.code==1]
        print(k,'pass TD yl>=40 /g',round(((p.tdp)&(p.yl>=40)).sum()/len(g),4),'TD per pass snap yl>=40',round(p[p.yl>=40].tdp.mean(),4),'snaps yl>=40 /g',round((p.yl>=40).sum()/len(g),3),'pass plays 40+ yds /g',round((p.yards>=40).sum()/len(g),4),'20+ /g',round((p.yards>=20).sum()/len(g),3),'pass TD/g',round(p.tdp.sum()/len(g),4),'run TD/g',round(D[(D.code==0)&D.tdp].shape[0]/len(g),4))
    a=pergame(R,(R.code==1)&R.tdp&(R.yl>=40),'tdp',gr);b=pergame(S,(S.code==1)&S.tdp&(S.yl>=40),'tdp',gs)
    e=boot(a,b,len(gr),len(gs));print('long pass TD/g sim-real',[round(x,4) for x in e])
    for nm,lo,hi in (('yl21-39',21,39),('yl<=20',1,20)):
        a=pergame(R,(R.code==1)&R.tdp&(R.yl>=lo)&(R.yl<=hi),'tdp',gr);b=pergame(S,(S.code==1)&S.tdp&(S.yl>=lo)&(S.yl<=hi),'tdp',gs)
        e=boot(a,b,len(gr),len(gs));print(nm,'pass TD/g sim-real',[round(x,4) for x in e])

def exact(R,S):
    gr=R.gid.unique();gs=S.gid.unique()
    print('=== exact yl 1-10 (all downs<=4): snaps/g, pass share, TD|pass, TD|run, total TD/snap, total TD/g real/sim')
    for y in range(1,11):
        o=[]
        for D,g in ((R,gr),(S,gs)):
            X=D[D.code.isin([0,1])&(D.yl==y)&(D.down<=4)]
            o.append((len(X)/len(g),(X.code==1).mean(),X[X.code==1].tdp.mean(),X[X.code==0].tdp.mean(),X.tdp.mean(),X.tdp.sum()/len(g)))
        print(y,' '.join(f'{a:.3f}/{b:.3f}' for a,b in zip(*o)))
    print('--- by down, yl 1-10: snaps/g, TD/snap real/sim')
    for d in (1,2,3,4):
        o=[]
        for D,g in ((R,gr),(S,gs)):
            X=D[D.code.isin([0,1])&(D.yl<=10)&(D.down==d)]
            o.append((len(X)/len(g),X.tdp.mean(),(X.code==1).mean()))
        print(d,' '.join(f'{a:.3f}/{b:.3f}' for a,b in zip(*o)))
    print('--- FG snaps and fourth-down go share at yl<=10')
    for D,g in ((R,gr),(S,gs)):
        X=D[(D.yl<=10)&(D.down==4)]
        print(len(X)/len(g),X.code.value_counts(normalize=True).round(3).to_dict())


if __name__=='__main__':
    R,S=load()
    print('games',R.gid.nunique(),S.gid.nunique())
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    if which in('seg','all'):seg(R,S)
    if which in('goal','all'):goal(R,S)
    if which in('exact','all'):exact(R,S)
