// ag4_proto.c: W-rank all-gather over RDMA WRITE-with-immediate, full mesh, host memory (step 1).
// Build: gcc -O2 -o ag4_proto ag4_proto.c -libverbs -lrdmacm
// Run:   ag4_proto <rank> <world> <chunk_bytes> <iters> <ip0> <ip1> ... (all ranks; port base 18515)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#define MAXW 8
#define PORT_BASE 18515
struct info { uint64_t addr; uint32_t rkey; uint32_t rank; };
static int rank, W, chunk, iters; static const char *ips[MAXW];
static struct ibv_pd *pd; static struct ibv_cq *cq; static struct ibv_mr *mr_recv, *mr_send;
static char *recvbuf, *sendbuf; static struct info my_info, peer_info[MAXW]; static struct rdma_cm_id *ids[MAXW];
static uint32_t qpn_of[MAXW];
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static void die(const char *m){ perror(m); exit(1); }
static void ensure_pd(struct ibv_context *ctx){
  if(pd) return; pd=ibv_alloc_pd(ctx); if(!pd) die("alloc_pd");
  cq=ibv_create_cq(ctx, 8192, NULL, NULL, 0); if(!cq) die("create_cq");
  recvbuf=aligned_alloc(4096,(size_t)W*chunk); sendbuf=aligned_alloc(4096,chunk);
  memset(recvbuf,0,(size_t)W*chunk); memset(sendbuf,'a'+rank,chunk);
  mr_recv=ibv_reg_mr(pd,recvbuf,(size_t)W*chunk,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE); if(!mr_recv) die("reg_mr recv");
  mr_send=ibv_reg_mr(pd,sendbuf,chunk,IBV_ACCESS_LOCAL_WRITE); if(!mr_send) die("reg_mr send");
  my_info.addr=(uint64_t)recvbuf; my_info.rkey=mr_recv->rkey; my_info.rank=rank;
}
static void make_qp(struct rdma_cm_id *id){
  ensure_pd(id->verbs);
  struct ibv_qp_init_attr qa={0}; qa.send_cq=cq; qa.recv_cq=cq; qa.qp_type=IBV_QPT_RC;
  qa.cap.max_send_wr=512; qa.cap.max_recv_wr=512; qa.cap.max_send_sge=1; qa.cap.max_recv_sge=1;
  if(rdma_create_qp(id,pd,&qa)) die("create_qp");
  for(int i=0;i<256;i++){ struct ibv_recv_wr wr={0},*bad; wr.wr_id=1; if(ibv_post_recv(id->qp,&wr,&bad)) die("post_recv"); }
}
static void expect(struct rdma_event_channel *ec, int want, struct rdma_cm_event **out){
  if(rdma_get_cm_event(ec,out)) die("get_cm_event");
  if((int)(*out)->event!=want){ fprintf(stderr,"rank%d expected event %d got %d\n",rank,want,(*out)->event); exit(1); }
}
int main(int argc,char**argv){
  if(argc<6){ fprintf(stderr,"usage\n"); return 1; }
  rank=atoi(argv[1]); W=atoi(argv[2]); chunk=atoi(argv[3]); iters=atoi(argv[4]); for(int i=0;i<W;i++) ips[i]=argv[5+i];
  struct rdma_event_channel *ec_in=rdma_create_event_channel(), *ec_out=rdma_create_event_channel();
  struct rdma_cm_id *lid=NULL;
  if(rank>0){ if(rdma_create_id(ec_in,&lid,NULL,RDMA_PS_TCP)) die("create_id");
    struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(PORT_BASE+rank); inet_pton(AF_INET,ips[rank],&a.sin_addr);
    if(rdma_bind_addr(lid,(struct sockaddr*)&a)) die("bind"); if(rdma_listen(lid,16)) die("listen"); }
  usleep(300000);
  // outbound to higher ranks
  for(int p=rank+1;p<W;p++){
    for(int tries=0;;tries++){
      struct rdma_cm_id *id; if(rdma_create_id(ec_out,&id,NULL,RDMA_PS_TCP)) die("create_id");
      struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_port=htons(PORT_BASE+p); inet_pton(AF_INET,ips[p],&a.sin_addr);
      struct sockaddr_in s={0}; s.sin_family=AF_INET; inet_pton(AF_INET,ips[rank],&s.sin_addr);
      struct rdma_cm_event *ev;
      if(rdma_resolve_addr(id,(struct sockaddr*)&s,(struct sockaddr*)&a,3000)) die("resolve_addr");
      expect(ec_out,RDMA_CM_EVENT_ADDR_RESOLVED,&ev); rdma_ack_cm_event(ev);
      if(rdma_resolve_route(id,3000)) die("resolve_route"); expect(ec_out,RDMA_CM_EVENT_ROUTE_RESOLVED,&ev); rdma_ack_cm_event(ev);
      make_qp(id);
      struct rdma_conn_param cp={0}; cp.responder_resources=8; cp.initiator_depth=8; cp.retry_count=7; cp.rnr_retry_count=7; cp.private_data=&my_info; cp.private_data_len=sizeof(my_info);
      if(rdma_connect(id,&cp)) die("connect");
      if(rdma_get_cm_event(ec_out,&ev)) die("get_cm_event");
      if(ev->event==RDMA_CM_EVENT_ESTABLISHED){ memcpy(&peer_info[p],ev->param.conn.private_data,sizeof(struct info)); rdma_ack_cm_event(ev); ids[p]=id; qpn_of[p]=id->qp->qp_num; break; }
      int t=ev->event; rdma_ack_cm_event(ev); rdma_destroy_qp(id); rdma_destroy_id(id);
      if(tries>300){ fprintf(stderr,"rank%d cannot connect to %d (event %d)\n",rank,p,t); exit(1);} usleep(100000);
    }
  }
  // inbound from lower ranks
  for(int k=0;k<rank;k++){
    struct rdma_cm_event *ev; expect(ec_in,RDMA_CM_EVENT_CONNECT_REQUEST,&ev);
    struct rdma_cm_id *id=ev->id; struct info pi; memcpy(&pi,ev->param.conn.private_data,sizeof(pi)); rdma_ack_cm_event(ev);
    make_qp(id);
    struct rdma_conn_param cp={0}; cp.responder_resources=8; cp.initiator_depth=8; cp.retry_count=7; cp.rnr_retry_count=7; cp.private_data=&my_info; cp.private_data_len=sizeof(my_info);
    if(rdma_accept(id,&cp)) die("accept");
    expect(ec_in,RDMA_CM_EVENT_ESTABLISHED,&ev); rdma_ack_cm_event(ev);
    ids[pi.rank]=id; peer_info[pi.rank]=pi; qpn_of[pi.rank]=id->qp->qp_num;
  }
  // rounds
  double best=1e9,sum=0; int done=0;
  for(int it=0; it<iters+50; it++){
    double t0=now_us();
    for(int p=0;p<W;p++){ if(p==rank) continue;
      struct ibv_sge sge={.addr=(uint64_t)sendbuf,.length=(uint32_t)chunk,.lkey=mr_send->lkey};
      struct ibv_send_wr wr={0},*bad; wr.wr_id=2; wr.sg_list=&sge; wr.num_sge=1; wr.opcode=IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags=IBV_SEND_SIGNALED;
      wr.imm_data=htonl(it); wr.wr.rdma.remote_addr=peer_info[p].addr+(uint64_t)rank*chunk; wr.wr.rdma.rkey=peer_info[p].rkey;
      if(ibv_post_send(ids[p]->qp,&wr,&bad)) die("post_send"); }
    int gr=0,gs=0; struct ibv_wc wc[16];
    while(gr<W-1||gs<W-1){ int n=ibv_poll_cq(cq,16,wc);
      for(int i=0;i<n;i++){ if(wc[i].status!=IBV_WC_SUCCESS){ fprintf(stderr,"rank%d wc status %d op %d\n",rank,wc[i].status,wc[i].opcode); exit(1);} 
        if(wc[i].opcode==IBV_WC_RECV_RDMA_WITH_IMM){ gr++; for(int p=0;p<W;p++) if(p!=rank&&qpn_of[p]==wc[i].qp_num){ struct ibv_recv_wr r={0},*bad; r.wr_id=1; ibv_post_recv(ids[p]->qp,&r,&bad); break; } }
        else gs++; } }
    double dt=now_us()-t0; if(it>=50){ sum+=dt; if(dt<best) best=dt; done++; }
  }
  int ok=1; for(int p=0;p<W;p++){ if(p==rank) continue; if(recvbuf[(size_t)p*chunk]!='a'+p||recvbuf[(size_t)p*chunk+chunk-1]!='a'+p) ok=0; }
  printf("rank%d W=%d chunk=%d: all-gather round avg %.1f us best %.1f us content_ok=%d\n",rank,W,chunk,sum/done,best,ok); fflush(stdout);
  return 0;
}
