#include <memory.h>
#include <Graph.h>
#include <lemon/list_graph.h>
#include <lemon/connectivity.h>
#include <lemon/euler.h>

using namespace lemon;

extern "C"
void findEulerGold(EulerVertex * h_ev,
		unsigned int * h_l,
		unsigned int * h_e,
		unsigned int vcount,
		EulerEdge * h_ee,
		unsigned int ecount,
		unsigned int kmerLength){

	//declare graph
	ListDigraph g;
	
	ListDigraph::Node * vertices=NULL;
	ListDigraph::ArcMap<unsigned int> arcIndex(g);	
	ListDigraph::Arc arc;
	
	vertices=new ListDigraph::Node[vcount];
	
	//translate graph Data Structure
	for(unsigned int i=0;i<vcount;i++){
		vertices[i]=g.addNode();
	}
	for (unsigned int j=0;j<ecount; j++){
		arc=g.addArc(vertices[h_ee[j].v1],vertices[h_ee[j].v2]);
		arcIndex[arc]=j;
	}

	
	//find component
	
	ListDigraph::NodeMap<int> components(g);
	
	int componentCount=stronglyConnectedComponents(g, components);
	
	//initialize component mark
	unsigned int * componentMark=(unsigned int  *)malloc(sizeof(unsigned int )* componentCount);
	memset(componentMark,0,(sizeof(unsigned int)*componentCount));
	
	//traverse each vertex 
	//check if the component has been toured?
	//if not ,get ET
	int ccount=0;
	unsigned int j=0;
	unsigned int p =0;
	while(ccount<componentCount){
		
		while(componentMark[components[vertices[j]]] ==1 && j<vcount){
			j++;
		}
		ccount++;
		componentMark[components[vertices[j]]]=1;	
		DiEulerIt<ListDigraph> et(g,vertices[j]);
		unsigned int srcE,dstE;
		srcE=ecount;
		dstE=ecount;

		if(et!=INVALID){
			srcE=arcIndex[et];
			while(++et!=INVALID){
				dstE=arcIndex[et];
				h_ee[srcE].s=dstE;
				srcE=dstE;
	/*			if(ccount>83){
					printf("p :%d\n",p++);
					if( p>= 595360)
					{ 
						printf("boo\n");
					}
				}
	*/		}
		}
/*	if(ccount>80)
	{
		int k=0;
		k++;
	}
*/	printf("ccount:%d\n",ccount);
	}
	free(componentMark);
	delete(vertices);
}
