#include "path.h"
#include <list>
#include <string>
#include <google/sparse_hash_map>
#include <google/dense_hash_map>
#include <iterator>

using namespace  std;
using  google::sparse_hash_map;
using  google::dense_hash_map;
PathSystem::PathSystem(unsigned int readCount ,unsigned int vertexCount,unsigned int edgeCount):/*vertexMap(vertexCount),*/
		readMap(readCount*2),edgeMap(edgeCount)
		{
	//do nothing
	readMap.set_empty_key(readCount*2+readCount);
	edgeMap.set_empty_key(make_pair(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF));
}


/*
 * 	dense_hash_map <KEY_T,list<unsigned int > > vertexMap;
 * 		list<pair<KEY_T,KEY_T> > reads;
 *
 * */
PathSystem::~PathSystem(){
//delete lists
	PathSystem::readMap.clear();
	PathSystem::edgeMap.clear();
	//PathSystem::vertexMap.clear();
}
void PathSystem::addRead(unsigned int readId){//not required

	//PathSystem::readMap[readId]=new list< pair<KEY_T,KEY_T> >();
}
void PathSystem::findPathPxy(KEY_T vin, KEY_T vmid, KEY_T vout, list<unsigned int> &pathList){

	pathList.clear();
	EDGE_TYPE x(vin,vmid);
	EDGE_TYPE y(vmid,vout);

	PATH_INDEX_LIST_TYPE & readList=PathSystem::edgeMap[x].second;
	for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
		EDGE_LIST_TYPE & path=PathSystem::readMap[*readIt];
		EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),x);
		if(edgeIt != path.end() ){
			++edgeIt;
			if(edgeIt != path.end() && *edgeIt==y ){
				 pathList.push_back(*readIt);
			}
		}
	}

	/*
	PATH_INDEX_LIST_TYPE readList=PathSystem::vertexMap[vmid];

	for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
		EDGE_LIST_TYPE path=PathSystem::readMap[*readIt];
		EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),x);
		if(edgeIt != path.end() ){
			++edgeIt;
			if(edgeIt != path.end() && *edgeIt==y ){
				 pathList.push_back(*readIt);
			}
		}
	}*/
}

void PathSystem::findPathAt(KEY_T vin, KEY_T vmid, list<unsigned int> &pathList){
	pathList.clear();
	EDGE_TYPE x(vin,vmid);
	PATH_INDEX_LIST_TYPE &readList=PathSystem::edgeMap[x].second;
	for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
		EDGE_LIST_TYPE &path=PathSystem::readMap[*readIt];
		//EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),x);
		 if(path.back()==x)
		 {
			 pathList.push_back(*readIt);
		 }

	}
	/*
	 PATH_INDEX_LIST_TYPE readList=PathSystem::vertexMap[vmid];
		for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
			EDGE_LIST_TYPE path=PathSystem::readMap[*readIt];
			//EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),x);
			 if(path.back()==x)
			 {
				 pathList.push_back(*readIt);
			 }

		}
*/
}

void PathSystem::findPathFrom(VERTEX_TYPE vmid, VERTEX_TYPE vout, list<unsigned int> &pathList){
	pathList.clear();
	EDGE_TYPE y(vmid,vout);

	PATH_INDEX_LIST_TYPE &readList=PathSystem::edgeMap[y].second;
		for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
			EDGE_LIST_TYPE &path =PathSystem::readMap[*readIt];
			//EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),y);
			 if( path.front()==y)
			 {
				 pathList.push_back(*readIt);
			 }

		}
	/*
	PATH_INDEX_LIST_TYPE readList=PathSystem::vertexMap[vmid];
		for(PATH_INDEX_LIST_TYPE::iterator readIt=readList.begin();readIt!=readList.end(); readIt++){
			EDGE_LIST_TYPE path =PathSystem::readMap[*readIt];
			//EDGE_LIST_TYPE::iterator edgeIt=find(path.begin(),path.end(),y);
			 if( path.front()==y)
			 {
				 pathList.push_back(*readIt);
			 }

		}
*/
}
void PathSystem::vertexExist(KEY_T v){

}
void PathSystem::addPathToRead(unsigned int  readId, VERTEX_TYPE v1, VERTEX_TYPE v2,EDGE_LABEL_TYPE & edgeLabel){


	PathSystem::readMap[readId].push_back(make_pair(v1,v2));
	//PathSystem::vertexMap[v1].push_back(readId);
	//PathSystem::vertexMap[v2].push_back(readId);

//	PathSystem::edgeMap[PathSystem::readMap[readId].back()]=make_pair(edgeLabel,readId);
	PathSystem::edgeMap[PathSystem::readMap[readId].back()].first=edgeLabel;
	PathSystem::edgeMap[PathSystem::readMap[readId].back()].second.push_back(readId);//=PathSystem::readMap[readId];
}
//sparse_hash_map <KEY_T,list<unsigned int> > vertexMap;
//	sparse_hash_map<unsigned int,list<pair<KEY_T,KEY_T> > > readMap;


void PathSystem::xyDetach(VERTEX_TYPE vin, VERTEX_TYPE vmid, VERTEX_TYPE vout){

	EDGE_TYPE x(vin,vmid);
	EDGE_TYPE y(vmid,vout);
	EDGE_TYPE z(vin,vout);
	PATH_INDEX_LIST_TYPE&  xPath=edgeMap[x].second;
	PATH_INDEX_LIST_TYPE & yPath=edgeMap[y].second;

	EDGE_LABEL_TYPE zLabel=edgeMap[x].first+edgeMap[y].first;

	PATH_INDEX_LIST_TYPE::iterator xpathIt=xPath.begin();
	EDGE_LIST_TYPE::iterator xpos,ypos;
	while( xpathIt!=xPath.end() ){
		xpos=	find(readMap[*xpathIt].begin(),readMap[*xpathIt].end(),x);
		if(xpos != readMap[*xpathIt].end())
		{

			if( *xpos != readMap[*xpathIt].back()){ // not in _x
				xpos++;
				if(xpos!=readMap[*xpathIt].end() && (*xpos)==y){	//xy
					//remove y
					//x=z
					//update readMap
					xpos--;
					xpos=readMap[*xpathIt].erase(xpos); //remove x
					xpos=readMap[*xpathIt].erase(xpos); //remove y
					readMap[*xpathIt].insert(xpos,z);

					//update vertex map
							//vertexMap[vmid].remove(*xpathIt);


					edgeMap[z].second.push_back(*xpathIt);
					edgeMap[z].first=zLabel;	//edgeLabel
					edgeMap[y].second.remove(*xpathIt);
					xpathIt=edgeMap[x].second.erase(xpathIt);  //<-- erase //update xpathIt
				}else {
					xpathIt++;
				}

			}else if(*xpos==readMap[*xpathIt].back()){		//in _x
				xpos=readMap[*xpathIt].erase(xpos); //remove x
				readMap[*xpathIt].insert(xpos,z);

				//update vertex map
						//vertexMap[vmid].remove(*xpathIt);
						//vertexMap[vout].push_back(*xpathIt);  //add y to _x

				edgeMap[z].second.push_back(*xpathIt);
				edgeMap[z].first=zLabel;	//edgeLabel
				xpathIt=edgeMap[x].second.erase(xpathIt);  //<-- erase //update xpathIt
			}else {
				xpathIt++;
			}
		}else {
			xpathIt++;
		}
	}
	PATH_INDEX_LIST_TYPE::iterator ypathIt=yPath.begin();
	while( ypathIt!=yPath.end() ){
		ypos=	find(readMap[*ypathIt].begin(),readMap[*ypathIt].end(),y);
				if(ypos == readMap[*ypathIt].begin())
				{
					//readmap
					ypos=readMap[*ypathIt].erase(ypos);		//remove y
					readMap[*ypathIt].insert(ypos,z);

					//vertexmap
						//vertexMap[vmid].remove(*ypathIt);
						//vertexMap[vin].push_back(*ypathIt);

					//edgemap
					edgeMap[z].second.push_back(*ypathIt);
					edgeMap[z].first=zLabel;	//edgeLabel
					ypathIt=edgeMap[y].second.erase(ypathIt);  //<-- erase //update xpathIt
				}else{
					ypathIt++;
				}
	}
}
