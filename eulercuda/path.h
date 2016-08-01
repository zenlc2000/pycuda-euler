
#ifndef PATH_H_
#define PATH_H_

#include<list>
#include<utility>
#include <boost/functional/hash.hpp>
#include <google/sparse_hash_map>
#include <google/dense_hash_map>
#include <iterator>
#include <string>
#include "common.h"


using google::sparse_hash_map;
using google::dense_hash_map;
using namespace std;/*
namespace std{
namespace tr1 {
    template<>
    struct hash<pair<U,V> pair> {
        std::size_t operator()(my_key_type const &key) {
            return whatever;
        }
    };
}
}

*/
typedef unsigned int VERTEX_INDEX_TYPE;

typedef unsigned int PATH_INDEX_TYPE;

typedef KEY_T	VERTEX_TYPE;

typedef pair<VERTEX_TYPE,VERTEX_TYPE> EDGE_TYPE;
typedef string EDGE_LABEL_TYPE;
typedef list<VERTEX_INDEX_TYPE> VERTEX_INDEX_LIST_TYPE;


typedef list<PATH_INDEX_TYPE> PATH_INDEX_LIST_TYPE;
typedef list<EDGE_TYPE >  EDGE_LIST_TYPE;
//typedef sparse_hash_map<VERTEX_TYPE, VERTEX_INDEX_TYPE> VERTEX_INDEX_MAP_TYPE;
typedef sparse_hash_map<PATH_INDEX_TYPE,list<pair<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE> > > CONSISTENCY_MAP_TYPE;
typedef sparse_hash_map<VERTEX_INDEX_TYPE,PATH_INDEX_LIST_TYPE > RESOLVE_MAP_TYPE;
typedef dense_hash_map< EDGE_TYPE,pair<EDGE_LABEL_TYPE,PATH_INDEX_LIST_TYPE >,boost::hash<EDGE_TYPE> > EDGE_MAP_TYPE;
typedef dense_hash_map<PATH_INDEX_TYPE,EDGE_LIST_TYPE> READ_MAP_TYPE;


class PathSystem {

public :
	PathSystem(unsigned int readCount, unsigned int vertexCount,unsigned int edgeCount);
	~PathSystem();
	void addRead(unsigned int readId);
	void addPathToRead(unsigned int readId, KEY_T v1, KEY_T v2, EDGE_LABEL_TYPE & edgeLabel);
	void findPathPxy(KEY_T vin, KEY_T vmid, KEY_T vout, list<unsigned int> &pathList);
	void findPathAt(KEY_T vin, KEY_T vmid, list<unsigned int> &pathList);
	void findPathFrom(KEY_T vmid, KEY_T vout, list<unsigned int> &pathList);
	void vertexExist(KEY_T v);
	void xyDetach(VERTEX_TYPE vin, VERTEX_TYPE vmid, VERTEX_TYPE vout);
public:
	//sparse_hash_map <VERTEX_TYPE,PATH_INDEX_LIST_TYPE > vertexMap;
	READ_MAP_TYPE readMap;
	EDGE_MAP_TYPE edgeMap;


};



#endif /* PATH_H_ */
