#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Z88 structure -> watertight surface STL (ASCII)
Eingabe:
  Node:  "<id> 3 x y z"
  Elem:  "<eid> 17"  + nächste Zeile "n1 n2 n3 n4" (TET4)
Vorgehen:
  - Für jedes TET4 die 4 Dreiecksflächen erzeugen
  - Innenflächen (Vorkommen==2) entfernen -> Außenhaut
  - Dreiecke über Adjazenz konsistent orientieren
  - Orientierung nach außen kippen (gegen Mesh-Zentrum geprüft)
  - STL schreiben
"""

import sys, math
from collections import defaultdict, deque

INPATH = "z88structure.txt"
OUT_STL = "surface.stl"

def read_z88_tet4(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        lines=[ln.strip() for ln in f if ln.strip()]
    nodes={}; tets=[]
    i=0
    while i<len(lines):
        t=lines[i].split()
        if len(t)>=5 and t[1]=='3':  # Node
            nid=int(float(t[0])); nodes[nid]=(float(t[2]),float(t[3]),float(t[4]))
            i+=1; continue
        if len(t)==2 and t[1]=='17':  # TET4-Block
            if i+1<len(lines):
                nds=[int(float(x)) for x in lines[i+1].split()]
                if len(nds)==4: tets.append(tuple(nds))
            i+=2; continue
        i+=1
    if not nodes or not tets: raise RuntimeError("Keine Nodes/TET4 gefunden.")
    return nodes, tets

def tri_normal(a,b,c):
    ax,ay,az=a; bx,by,bz=b; cx,cy,cz=c
    ux,uy,uz=bx-ax,by-ay,bz-az; vx,vy,vz=cx-ax,cy-ay,cz-az
    nx=uy*vz-uz*vy; ny=uz*vx-ux*vz; nz=ux*vy-uy*vx
    n=math.sqrt(nx*nx+ny*ny+nz*nz) or 1.0
    return (nx/n,ny/n,nz/n)

def build_boundary_tris(tets):
    count=defaultdict(int); any_tri={}
    for a,b,c,d in tets:
        for tri in ((a,b,c),(a,b,d),(a,c,d),(b,c,d)):
            key=tuple(sorted(tri)); count[key]+=1; any_tri[key]=tri
    return [any_tri[k] for k,v in count.items() if v==1]

def orient_tris_consistently(tris):
    adj=defaultdict(list)
    for fi,(a,b,c) in enumerate(tris):
        for e in ((a,b),(b,c),(c,a)):
            adj[tuple(sorted(e))].append(fi)
    def has_dir(t,u,v):
        return (t[0]==u and t[1]==v) or (t[1]==u and t[2]==v) or (t[2]==u and t[0]==v)
    out=[None]*len(tris)
    for s in range(len(tris)):
        if out[s] is not None: continue
        out[s]=list(tris[s]); q=deque([s])
        while q:
            f=q.popleft(); a,b,c=out[f]
            for (u,v) in ((a,b),(b,c),(c,a)):
                key=tuple(sorted((u,v)))
                for g in adj[key]:
                    if g==f or out[g] is not None: continue
                    t=list(tris[g])
                    if has_dir(t,u,v): t=[t[0],t[2],t[1]]  # flip
                    out[g]=t; q.append(g)
    return [tuple(t) for t in out]

def flip_outward(verts, tris):
    cx=sum(verts[i][0] for i in verts)/len(verts)
    cy=sum(verts[i][1] for i in verts)/len(verts)
    cz=sum(verts[i][2] for i in verts)/len(verts)
    score=0.0
    for a,b,c in tris:
        va,vb,vc=verts[a],verts[b],verts[c]
        nx,ny,nz=tri_normal(va,vb,vc)
        mx=(va[0]+vb[0]+vc[0])/3 - cx
        my=(va[1]+vb[1]+vc[1])/3 - cy
        mz=(va[2]+vb[2]+vc[2])/3 - cz
        score += nx*mx + ny*my + nz*mz
    return [(a,c,b) for (a,b,c) in tris] if score<0 else tris

def write_stl_ascii(path, verts, tris):
    with open(path,'w',encoding='utf-8') as f:
        f.write("solid z88_surface\n")
        for a,b,c in tris:
            va,vb,vc=verts[a],verts[b],verts[c]
            nx,ny,nz=tri_normal(va,vb,vc)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n    outer loop\n")
            for v in (va,vb,vc):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid z88_surface\n")

def main(inpath=INPATH, out_stl=OUT_STL):
    nodes, tets = read_z88_tet4(inpath)
    tris = build_boundary_tris(tets)
    tris = orient_tris_consistently(tris)
    tris = flip_outward(nodes, tris)

    # Quick manifold-check
    edgecnt=defaultdict(int)
    for a,b,c in tris:
        for e in ((a,b),(b,c),(c,a)):
            edgecnt[tuple(sorted(e))]+=1
    e1=sum(1 for v in edgecnt.values() if v==1)
    e2=sum(1 for v in edgecnt.values() if v==2)
    egt=sum(1 for v in edgecnt.values() if v>2)
    print(f"Surface faces: {len(tris)} | edges 1:{e1} 2:{e2} >2:{egt}")

    if e1 or egt:
        print("WARN: Oberfläche ist nicht streng 2-mannigfaltig (evtl. offen oder mehrfach belegt).")

    write_stl_ascii(out_stl, nodes, tris)
    print(f"STL geschrieben: {out_stl}")

if __name__ == "__main__":
    if len(sys.argv)>=2: INPATH=sys.argv[1]
    if len(sys.argv)>=3: OUT_STL=sys.argv[2]
    main(INPATH, OUT_STL)
