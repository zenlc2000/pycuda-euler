#include "Graph.h"
#include "path.h"
#include <google/sparse_hash_map>



using  google::sparse_hash_map;
using namespace std;

void validateGraph(EulerVertex * ev, EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int& vertexCount,unsigned int& edgeCount){

	//validate lev ent
	unsigned int lcount=0;
	unsigned int ecount=0;

	printf("checking lev/ent edges ...\n");
	for(unsigned int i=0;i<edgeCount; i++){
		if(levEdge[i]>=edgeCount){
			lcount++;
		}
	}
	for(unsigned int i=0;i<edgeCount; i++){
			if(entEdge[i]>=edgeCount){
				ecount++;
			}
	}
	printf("levEdge has %d invalid edges, entEdge has %d invalid edges \n",lcount,ecount);

	for(unsigned int i=0;i<vertexCount;i++){
		lcount=0;ecount=0;
		for(unsigned int base=ev[i].lp,  j=0; j<ev[i].lcount;j++){
			if(levEdge[base+j]>=edgeCount)
				lcount++;
		}
		for(unsigned int base=ev[i].ep, j=0; j<ev[i].ecount;j++){
			if(entEdge[base+j]>=edgeCount)
				ecount++;
		}
		if(lcount>0 || ecount>0){
			printf("invalid vertex %d , ecount:%d, lcount:%d\n",i,ecount,lcount);
		}

	}

	ecount=0;lcount=0;
	for(unsigned int i =0; i<edgeCount;i++){
		if(ee[i].eid>=edgeCount)			{
			ecount++;
			printf("Invalid edge %d\n",i);
			}
	}

	printf("Total invalid edges %d\n",ecount);

}


void applycompression() {

}

unsigned int getlev(VERTEX_INDEX_TYPE v1, VERTEX_INDEX_TYPE v2, EulerVertex * ev,
		EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int vertexCount, unsigned int edgeCount){
	unsigned int index=edgeCount;
	index=ev[v1].lp;
	while (index<ev[v1].lp+ev[v1].lcount && (ee[levEdge[index]].v2!=v2) ) index++;
	if( index<edgeCount &&  (ee[levEdge[index]].v2==v2)) return index;
	else return edgeCount;

}
unsigned int getent(VERTEX_INDEX_TYPE v1, VERTEX_INDEX_TYPE v2, EulerVertex * ev,

		EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int vertexCount, unsigned int edgeCount){
	unsigned int index=edgeCount;
	index=ev[v2].ep;
	while (index<ev[v2].ep+ev[v2].ecount && (ee[entEdge[index]].v1!=v1) ) index++;
	if(index<edgeCount && ee[entEdge[index]].v1==v1) return index;
	else return edgeCount;
}

void checkEmpty(EulerVertex * ev,unsigned int * levEdge, unsigned int * entEdge, unsigned int v, unsigned int edgeCount){
	unsigned int lcount=0; unsigned int ecount=0;

		//debug
		lcount=0;ecount=0;
		for(unsigned int base=ev[v].lp,  j=0; j<ev[v].lcount;j++){
				if(levEdge[base+j]>=edgeCount)
					lcount++;
			}
			for(unsigned int base=ev[v].ep, j=0; j<ev[v].ecount;j++){
				if(entEdge[base+j]>=edgeCount)
					ecount++;
			}
			if(lcount>0 || ecount>0){
				printf("invalid vertex %d , ecount:%d, lcount:%d\n",v,ecount,lcount);
			}

}
bool applyXYDetachment(VERTEX_INDEX_TYPE vin, VERTEX_INDEX_TYPE vmid, VERTEX_INDEX_TYPE vout, PathSystem & P, EulerVertex * ev,
		EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int vertexCount, unsigned int edgeCount) {



	unsigned int xIdx;
	unsigned int vinlIdx;
	unsigned int vmideIdx;
	unsigned int vmidlIdx;
	unsigned int vouteIdx;
	unsigned int xcount;
	unsigned int yIdx;
	unsigned int index;
	unsigned int ycount;


	///debug
	//for(unsigned int t=0;t<vertexCount;t++){
	//checkEmpty(ev,levEdge,entEdge,t,edgeCount);
//	}
	//checkEmpty(ev,levEdge,entEdge,vin,edgeCount);
	//checkEmpty(ev,levEdge,entEdge,vmid,edgeCount);
	//checkEmpty(ev,levEdge,entEdge,vout,edgeCount);
	//x info

	//vin-vmid
	index=getlev(vin,vmid,ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	if(index<edgeCount) {
		xIdx=levEdge[index];
		vinlIdx=index;
	}
	xcount=0;
	index=xIdx;
	while(ee[index].v1== vin  && ee[index].v2==vmid && index<edgeCount) {xcount++; index++;} 	//get number of edges;

	//find vmid index;

	index=getent(vin,vmid,ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	if(index<edgeCount) vmideIdx=index;

	//y
	//vmid - vin
	index=getlev(vmid,vout,ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	if(index<edgeCount) {
		yIdx=levEdge[index];
		vmidlIdx=index;
	}
	ycount=0;
	index=yIdx;
	while(ee[index].v1== vmid  && ee[index].v2==vout && index<edgeCount) {ycount++; index++;} 	//get number of edges;

	index=getent(vmid,vout,ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	if(index<edgeCount) vouteIdx=index;

	unsigned j=0;
	unsigned int netCount=xcount<ycount?xcount:ycount;
	for(;j<netCount; j++){

		//vin
		levEdge[vinlIdx]=xIdx;

		//vmid
		entEdge[vmideIdx]=edgeCount;
		levEdge[vmidlIdx]=edgeCount;

		//vout
		entEdge[vouteIdx]=xIdx;		//x is new z

		//x
		ee[xIdx].v2=vout;		//modify x to be new z
		//y	kill
		ee[yIdx].eid=edgeCount;
		ee[yIdx].v1=vertexCount;
		ee[yIdx].v2=vertexCount;

		//x


			xIdx++;
			yIdx++;
			vinlIdx++;
			vmidlIdx++;
			vmideIdx++;
			vouteIdx++;

	}
	if(netCount>0){
		//ent lev ev.ecount ev.lcount;
		if(netCount<ev[vmid].ecount && vmideIdx<ev[vmid].ecount+ev[vmid].ep){
			unsigned int moveCount=(ev[vmid].ecount+ev[vmid].ep)-vmideIdx;
			for(j=0;j<moveCount ;j++){

					entEdge[vmideIdx-netCount+j]=entEdge[vmideIdx+j];
					entEdge[vmideIdx+j]=edgeCount;
				}

		}
		if( netCount<ev[vmid].lcount && vmidlIdx<ev[vmid].lcount+ev[vmid].lp){
			//for(j=0;j<netCount && j< ev[vmid].lcount-netCount ;j++){
			unsigned int moveCount=(ev[vmid].lcount+ev[vmid].lp)-vmidlIdx;
			for(j=0;j<moveCount ;j++){
				levEdge[vmidlIdx-netCount+j]=levEdge[vmidlIdx+j];
				levEdge[vmidlIdx+j]=edgeCount;
				}

		}

		ev[vmid].ecount-=netCount;
		ev[vmid].lcount-=netCount;

		P.xyDetach(ev[vin].vid,ev[vmid].vid,ev[vout].vid);

	}


	////debug
//	for(unsigned int t=0;t<vertexCount;t++){
	//checkEmpty(ev,levEdge,entEdge,t,edgeCount);
	//}
//	checkEmpty(ev,levEdge,entEdge,vin,edgeCount);
//		checkEmpty(ev,levEdge,entEdge,vmid,edgeCount);
//		checkEmpty(ev,levEdge,entEdge,vout,edgeCount);
	///debug

	return netCount>0;
}
void applyXCut() {

}
 void getVin(EulerVertex & vmid, EulerVertex * ev, unsigned int * levEdge, unsigned int * entEdge, EulerEdge * ee,
		unsigned int vertexCount, unsigned int edgeCount,  VERTEX_INDEX_LIST_TYPE &vin) {

	//VERTEX_INDEX_TYPE vmidIdx= vertexIndexMap[vmid.vid];
	vin.clear();


	for (unsigned int i = 0; i < vmid.ecount; i++) {
		if (entEdge[vmid.ep + i] < edgeCount) {

			if(find(vin.begin(),vin.end(),ee[entEdge[vmid.ep + i]].v1) ==vin.end())
				vin.push_back(ee[entEdge[vmid.ep + i]].v1);
			}
	}

}
void  getVout(EulerVertex & vmid, EulerVertex * ev, unsigned int * levEdge, unsigned int * entEdge, EulerEdge * ee,
		unsigned int vertexCount, unsigned int edgeCount,   VERTEX_INDEX_LIST_TYPE &vout) {

	//VERTEX_INDEX_TYPE vmidIdx= vertexIndexMap[vmid.vid];
	vout.clear();

	for (unsigned int i = 0; i < vmid.lcount; i++) {
			if (entEdge[vmid.lp + i] < edgeCount) {

				if(find(vout.begin(),vout.end(),ee[levEdge[vmid.lp + i]].v2) ==vout.end())
					vout.push_back(ee[levEdge[vmid.lp + i]].v2);
				}
		}


}

void findPathAt(EulerVertex &vin,
				EulerVertex &vmid,
				PathSystem & P,
				/*out*/
						PATH_INDEX_LIST_TYPE & P_x
		){

	P.findPathAt(vin.vid,vmid.vid,P_x);
}
void findPathPxy(EulerVertex & vin,
						EulerVertex & vmid,
						EulerVertex &vout,
						PathSystem &P,
						/* out */
						PATH_INDEX_LIST_TYPE & Pxy
		){


	P.findPathPxy(vin.vid,vmid.vid,vout.vid,Pxy);

}

bool isConsistentPath(PathSystem & P,PATH_INDEX_TYPE p1, PATH_INDEX_TYPE p2, VERTEX_TYPE v1, VERTEX_TYPE v2){

	bool consistent=false;
	EDGE_TYPE x(v1,v2);
	list<pair<VERTEX_TYPE,VERTEX_TYPE> > read1=P.readMap[p1];
	list<pair<VERTEX_TYPE, VERTEX_TYPE > > read2=P.readMap[p2];

	list<pair<VERTEX_TYPE,VERTEX_TYPE> >::iterator it1=find(read1.begin(),read1.end(),x);

	list<pair<VERTEX_TYPE,VERTEX_TYPE> >::iterator it2=find(read2.begin(),read2.end(),x);
	while(it1!=read1.end() && it2!=read2.end() && *it1==*it2){
		it1++;
		it2++;
	}
	if(it1 == read1.end() || it2 == read2.end())
		consistent=true;
	else
		consistent=false;
	if(consistent){
	//check backward
		it1=find(read1.begin(),read1.end(),x);
		it2=find(read2.begin(),read2.end(),x);

		if(it1!=read1.end() && it2!=read2.end()){
			while(it1!=read1.begin() && it2!=read2.begin() && *it1==*it2){
				it1--;
				it2--;
			}
			if(*it1==*it2)
				consistent=true;
			else
				consistent=false;
		}

	}

	return consistent;
}
bool isConsistentPathSet(PathSystem &P,PATH_INDEX_TYPE Px, PATH_INDEX_LIST_TYPE &Pxy, VERTEX_TYPE v1, VERTEX_TYPE v2){

	for(PATH_INDEX_LIST_TYPE::iterator it=Pxy.begin(); it!=Pxy.end(); it++){
		if(!isConsistentPath(P,Px,*it,v1,v2))
			return false;
	}
	return true;
}
bool resolved(CONSISTENCY_MAP_TYPE &consistency,PathSystem & P, PATH_INDEX_LIST_TYPE & P_x, VERTEX_INDEX_TYPE vin, VERTEX_INDEX_TYPE vmid ,EulerVertex * ev,
		EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int vertexCount, unsigned int edgeCount){

	RESOLVE_MAP_TYPE resolveMap;
	bool isResolved=false;
	for (CONSISTENCY_MAP_TYPE::iterator pxIdx=consistency.begin(); pxIdx!=consistency.end(); pxIdx++){
		for (list<pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE> > ::iterator pyIdx= (*pxIdx).second.begin(); pyIdx!=(*pxIdx).second.end(); pyIdx++){
			resolveMap[(*pyIdx).first].push_back((*pxIdx).first);
			if(resolveMap.size()>1)	break;
		}
		if(resolveMap.size()>1)	break; //short circuit
	}

	isResolved=(resolveMap.size()==1);
	if(isResolved){
		return applyXYDetachment(vin,vmid,(*(resolveMap.begin())).first,P,ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	}
	return isResolved;
}
void fixGraph(EulerVertex * ev, EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int& vertexCount,unsigned int& edgeCount		){

	typedef dense_hash_map<unsigned int,unsigned int> EDGE_INDEX_MAP_TYPE;
	EDGE_INDEX_MAP_TYPE edgeIndexMap(edgeCount);
	edgeIndexMap.set_empty_key(edgeCount);
	list<EulerEdge> edgeList=list<EulerEdge> ();
	unsigned int filterEdgeCount=0;

	validateGraph(ev,ee,levEdge,entEdge,vertexCount,edgeCount);

	for(unsigned int i=0;i<edgeCount; i++){
		if(ee[i].eid<edgeCount){

			edgeIndexMap[i]=filterEdgeCount;
			filterEdgeCount++;
			edgeList.push_back(ee[i]);
		}
	}
	unsigned int j=0;
	for(list<EulerEdge>::iterator edgeListIt=edgeList.begin();edgeListIt!=edgeList.end();edgeListIt++){
		ee[j]=*edgeListIt;
		j++;
	}
	for(unsigned int k=0;k<vertexCount;k++){
		for(unsigned int p=ev[k].lp; p<ev[k].lp+ev[k].lcount; p++){
			levEdge[p]=edgeIndexMap[levEdge[p]];
		}
		for(unsigned int q=ev[k].ep; q<ev[k].ep+ev[k].ecount; q++){
					entEdge[q]=edgeIndexMap[entEdge[q]];
				}
	}
	edgeCount=filterEdgeCount;

	validateGraph(ev,ee,levEdge,entEdge,vertexCount,edgeCount);
}
void checkGraph(EulerVertex * ev, EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int& vertexCount,unsigned int& edgeCount		){

	KEY_T lmerMask[] ={
	    0x0000000000000003, 0x000000000000000F, 0x000000000000003F, 0x00000000000000FF, // 0   1   2   3
	    0x00000000000003FF, 0x0000000000000FFF, 0x0000000000003FFF, 0x000000000000FFFF, // 4   5   6   7
	    0x000000000003FFFF, 0x00000000000FFFFF, 0x00000000003FFFFF, 0x0000000000FFFFFF, // 8   9   10  11
	    0x0000000003FFFFFF, 0x000000000FFFFFFF, 0x000000003FFFFFFF, 0x00000000FFFFFFFF, // 12  13  14  15
	    0x00000003FFFFFFFF, 0x0000000FFFFFFFFF, 0x0000003FFFFFFFFF, 0x000000FFFFFFFFFF, // 16  17  18  19
	    0x000003FFFFFFFFFF, 0x00000FFFFFFFFFFF, 0x00003FFFFFFFFFFF, 0x0000FFFFFFFFFFFF, // 20  21  22  23
	    0x0003FFFFFFFFFFFF, 0x000FFFFFFFFFFFFF, 0x003FFFFFFFFFFFFF, 0x00FFFFFFFFFFFFFF, // 24  25  26  27
	    0x03FFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF // 28  29  30  31
	};

	printf(" bit mask %lu \n",lmerMask[18]);
	for(unsigned int i=0;i<edgeCount;i++){
		if(ee[i].v1>=vertexCount || ee[i].v2>=vertexCount){
			printf("edge %lu lmer:%lu v1 %lu, v2 %lu",i,ee[i].eid,ee[i].v1,ee[i].v2);
			printf(" should be v1:%lu v2:%lu",LMER_PREFIX(ee[i].eid,lmerMask[18]),LMER_SUFFIX(ee[i].eid,lmerMask[18]));
			if(ee[i].v1<vertexCount ) printf("  [v1.vid=> %lu]",ev[ee[i].v1].vid);
			if(ee[i].v2<vertexCount ) printf("  [v2.vid=> %lu]",ev[ee[i].v2].vid);
			printf ("\n");
		}
	}
}
extern "C"
void transform(EulerVertex * ev, EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int& edgeCount,
		unsigned int& vertexCount, PathSystem & P, char * reads, unsigned int readCount) {

	bool changed = true;
	EulerVertex vmid;
	VERTEX_INDEX_LIST_TYPE vin, vout;


	PATH_INDEX_LIST_TYPE P_x;
	PATH_INDEX_LIST_TYPE Py_;
	//list<unsigned int> Pxy;
	list<pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE > > PxyList;
	//PATH_INDEX_LIST_TYPE * pxy;
	VERTEX_INDEX_TYPE vmidIdx,vinIdx;

	CONSISTENCY_MAP_TYPE consistencyMap;
	//VERTEX_INDEX_MAP_TYPE vertexIndexMap(vertexCount);
	unsigned int vzero = vertexCount;

	//vertexIndexMap.set_empty_key((KEY_T)(0));
	//construct vertex map
	//for (unsigned int i = 0; i < vertexCount; i++) {
//		vertexIndexMap[ev[i].vid] = i;
//	}
	checkGraph(ev,ee,levEdge,entEdge,vertexCount,edgeCount);
//	return;
	while (changed) {
		changed = false;
		//for (sparse_hash_map<VERTEX_TYPE,PATH_INDEX_LIST_TYPE >::iterator i = P.vertexMap.begin(); i != P.vertexMap.end(); i++) {
		for (VERTEX_INDEX_TYPE i = 0; i <vertexCount;i++) {
			vmid = ev[i];
			vmidIdx=i;//vertexIndexMap[(*i).first];

			getVin(vmid, ev, levEdge, entEdge, ee, vertexCount, edgeCount,  vin);
			getVout(vmid, ev, levEdge, entEdge, ee, vertexCount, edgeCount, vout);
			consistencyMap.clear();
			for (VERTEX_INDEX_LIST_TYPE::iterator   vinIt= vin.begin(); vinIt!=vin.end(); vinIt++) {
				//x= vin-vmid, y= vmid-vout
				EulerVertex vx = ev[*vinIt];	//<--
				vinIdx=*vinIt;

				findPathAt(vx, vmid, P, P_x);
				PxyList.clear();
				for (VERTEX_INDEX_LIST_TYPE::iterator voutIt = vout.begin(); voutIt !=vout.end(); voutIt++) {
					//check consisteny Pxi, Pyi

					pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE > entry;
					entry.first=*voutIt;
					PxyList.push_back(entry);
					EulerVertex vy = ev[*voutIt];
					findPathPxy(vx, vmid, vy,P, PxyList.back().second);
				}
				consistencyMap.clear();
				if(P_x.begin()!=P_x.end()){ //P_x not empty


					//all read in P_x  must be resolvable i.e consistent with exactly one Pxy edge
					for (PATH_INDEX_LIST_TYPE::iterator P_xIt = P_x.begin(); P_xIt!=P_x.end(); P_xIt++) {
						for (list<pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE > >::iterator PxyListIt = PxyList.begin(); PxyListIt !=PxyList.end(); PxyListIt++) {
							if(isConsistentPathSet(P,*P_xIt, (*PxyListIt).second,vx.vid,vmid.vid))
								consistencyMap[*P_xIt].push_back(*PxyListIt);
						}
					}
					if(resolved(consistencyMap,P,P_x,vinIdx,vmidIdx,ev,ee,levEdge,entEdge,vertexCount,edgeCount)){
						changed = true;
					}
				}else {
					for (list<pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE > >::iterator PxyListIt = PxyList.begin(); PxyListIt !=PxyList.end(); PxyListIt++) {
						if( applyXYDetachment(vinIdx,vmidIdx,(*PxyListIt).first,P,ev,ee,levEdge,entEdge,vertexCount,edgeCount) ){
							changed=true;
						}
					}
				}
			}
		}
	}
	//find a spot

	//apply transform

	//find a spot
	//apply cut
	fixGraph(ev,ee,levEdge,entEdge,vertexCount,edgeCount);
	consistencyMap.clear();
}
