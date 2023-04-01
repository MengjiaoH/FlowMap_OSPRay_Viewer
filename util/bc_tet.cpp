#include "bc_tet.h"

void crossProduct(vec3f &ans, vec3f &v1, vec3f &v2)
{
  ans.x = v1.y*v2.z - v1.z*v2.y;
  ans.y = v1.z*v2.x - v1.x*v2.z;
  ans.z = v1.x*v2.y - v1.y*v2.x;
  return;
}

double dotProduct( vec3f &v1, vec3f &v2 )
{
  double result = v1.x*v2.x+ v1.y*v2.y + v1.z*v2.z;
  return (result);
}


void bary_tet( vec4f &ans, vec3f &p, vec3f &a, vec3f &b, vec3f &c, vec3f &d )
{
    vec3f vap;
    vec3f vbp;
    vec3f vcp;
    vec3f vdp;
    vec3f vab;
    vec3f vac;
    vec3f vad;
    vec3f vbc;
    vec3f vbd;
    float va;
    float vb;
    float vc;
    float vd;
    float v;
    vec3f temp;

    vap = p - a;
    vbp = p - b;
    vcp = p - c;
    vdp = p - d;
    vab = b - a;
    vac = c - a;
    vad = d - a;
    vbc = c - b;
    vbd = d - b;

  crossProduct( temp, vbd, vbc );
  va = dotProduct( vbp, temp ) / 6.0 ;
  crossProduct( temp, vac, vad );
  vb = dotProduct( vap, temp ) / 6.0 ;
  crossProduct( temp, vad, vab );
  vc = dotProduct( vap, temp ) / 6.0 ;
  crossProduct( temp, vab, vac );
  vd = dotProduct( vap, temp ) / 6.0 ;
  crossProduct( temp, vac, vad );
  v  = dotProduct( vab, temp ) / 6.0 ;

  ans.x = va / v;
  ans.y = vb / v;
  ans.z = vc / v;
  ans.w = vd / v;
  
  return;
}