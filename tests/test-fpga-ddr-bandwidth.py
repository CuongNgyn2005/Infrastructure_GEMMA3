#!/usr/bin/env python3
"""Compile extracted production APM policy/lifecycle against RAM MMIO and clocks."""
import os
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / 'tools/fpga-ddr-bandwidth.c').read_text()
def section(first, last):
    begin = source.index(first)
    return source[begin:source.index(last, begin)]
headers = r'''
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
'''
code = headers + section('#define DDR_APM_PHYS_ADDR', 'static void usage(')
code += r'''
static char output[32768];
static size_t used;
static int print_fail, print_fail_at, print_calls, flush_fail_at, flush_calls;
static int mock_printf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
static int mock_printf(const char * fmt, ...) {
    va_list ap; int n;
    if (print_fail || ++print_calls == print_fail_at) return -1;
    va_start(ap,fmt); n=vsnprintf(output+used,sizeof(output)-used,fmt,ap); va_end(ap);
    assert(n>=0 && (size_t)n < sizeof(output)-used); used+=(size_t)n; return n;
}
#define printf mock_printf
static int mock_flush(FILE * f) { (void)f; return ++flush_calls == flush_fail_at ? -1 : 0; }
#define fflush mock_flush
'''
code += section('static void usage(', 'static int read_text(')
code += r'''
static uint32_t ram[0x400/4], params[10];
static unsigned starts, stops, reads, writes, selector_writes;
static unsigned read_offsets[100];
static int selector_fail, signal_setup, sleep_fail, clock_fail, clock_calls, bad_time, interrupt_sleep;
static uint32_t inject_status;
static double fake_time, last_active_begin;
static int ever_started;
static uint32_t mmio_read(volatile uint32_t * regs, uint32_t off) {
    if (ever_started && (off==REG_INTERRUPT_STATUS || (off>=0x100 && off<=0x190)))
        assert(!(regs[REG_CONTROL/4]&CTRL_METRIC_ENABLE));
    if (off>=0x100 && off<=0x190) { assert(reads<100); read_offsets[reads++]=off; }
    if (selector_fail && off==REG_METRIC_SELECTOR1 && selector_writes>=2) return regs[off/4]^1U;
    return regs[off/4];
}
static void mmio_write(volatile uint32_t * regs, uint32_t off, uint32_t value) {
    ++writes;
    if (off>=REG_METRIC_SELECTOR0 && off<=REG_METRIC_SELECTOR2) {
        assert(!(regs[REG_CONTROL/4]&CTRL_METRIC_ENABLE)); ++selector_writes;
        if (signal_setup) on_signal(SIGINT);
    }
    if (off==REG_INTERRUPT_STATUS) { regs[off/4]&=~value; return; }
    if (off==REG_CONTROL) {
        if (value&CTRL_METRIC_RESET) for (unsigned i=0;i<10;++i) regs[(0x100+16*i)/4]=0;
        if (value&CTRL_METRIC_ENABLE) {
            ++starts; ever_started=1;
            for (unsigned i=0;i<10;++i) regs[(0x100+16*i)/4]=UINT32_MAX;
        } else if (regs[off/4]&CTRL_METRIC_ENABLE) { ++stops; regs[REG_INTERRUPT_STATUS/4]=inject_status; }
    }
    regs[off/4]=value;
    /* The real mmio_write performs this readback before returning. */
    (void)mmio_read(regs,off);
}
static int mock_clock(int kind, struct timespec * out) {
    (void)kind; ++clock_calls;
    if (clock_fail==clock_calls) return -1;
    double value=fake_time;
    if (clock_calls==4) last_active_begin=value;
    if (clock_calls==5 && bad_time==1) value=last_active_begin;
    if (clock_calls==5 && bad_time==2) value=last_active_begin-0.01;
    if (clock_calls==6 && bad_time==3) value=last_active_begin;
    if (clock_calls==8 && bad_time==4) value=0.01;
    out->tv_sec=(time_t)value; out->tv_nsec=(long)((value-(double)out->tv_sec)*1e9);
    fake_time+=0.001; return 0;
}
static int mock_sleep(const struct timespec * req, struct timespec * rem) {
    (void)rem; assert(req->tv_sec==0 && req->tv_nsec<=100000000L);
    fake_time+=(double)req->tv_nsec/1e9;
    if (interrupt_sleep) on_signal(SIGTERM);
    if (sleep_fail) { errno=EINVAL; return -1; }
    return 0;
}
#define clock_gettime mock_clock
#define nanosleep mock_sleep
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#define CLOCK_REALTIME 0
#endif
'''
code += section('static int prepare_counter_window(', 'static int acquire_lock(')
code += section('static int check_capabilities(', 'static int print_wall_timestamp(')
code += 'static int print_wall_timestamp(const struct timespec * t) { (void)t; return 0; }\n'
code += section('static int run_measurement(', '\nint main(')
code += r'''
static int map_fail, unmapped;
static int map_apm(const char * path, struct mapping * mapping) {
    (void)path; mapping->regs=ram; mapping->params=params; return map_fail ? -1 : 0;
}
static void unmap_apm(struct mapping * mapping) { (void)mapping; ++unmapped; }
static int lifecycle(struct options options) {
    struct mapping mapping={-1,0,0,NULL,NULL}; int result=1;
'''
code += section('    if (map_apm(options.uio_path', '\n}', ) + '\n}\n'
code += r'''
static void reset(void) {
    memset(ram,0,sizeof(ram)); memset(&saved,0,sizeof(saved));
    ram[REG_METRIC_SELECTOR0/4]=0xdeadbeef; ram[REG_METRIC_SELECTOR1/4]=0x11223344;
    ram[REG_METRIC_SELECTOR2/4]=0x55667788; ram[REG_CONTROL/4]=CTRL_WRITE_LAT_START;
    ram[REG_SAMPLE_INTERVAL_HI/4]=7; ram[REG_SAMPLE_INTERVAL_LO/4]=9;
    ram[REG_SAMPLE_INTERVAL_CTL/4]=SICR_METRIC_RESET;
    params[0]=1;params[1]=6;params[2]=1;params[5]=10;params[6]=32;params[9]=1;
    used=0;output[0]=0;starts=stops=reads=writes=selector_writes=0;
    print_fail_at=print_calls=flush_fail_at=flush_calls=0;
    print_fail=selector_fail=signal_setup=sleep_fail=clock_fail=clock_calls=bad_time=interrupt_sleep=0;
    stop_requested=0;inject_status=0;fake_time=0;ever_started=0;map_fail=unmapped=0;
}
static void restored(void) {
    assert(ram[REG_METRIC_SELECTOR0/4]==0xdeadbeef && ram[REG_METRIC_SELECTOR1/4]==0x11223344);
    assert(ram[REG_METRIC_SELECTOR2/4]==0x55667788 && ram[REG_CONTROL/4]==CTRL_WRITE_LAT_START);
    assert(ram[REG_SAMPLE_INTERVAL_HI/4]==7 && ram[REG_SAMPLE_INTERVAL_LO/4]==9);
    assert(ram[REG_SAMPLE_INTERVAL_CTL/4]==SICR_METRIC_RESET);
    assert(ram[REG_INTERRUPT_ENABLE/4]==0 && ram[REG_GLOBAL_INTERRUPT/4]==0 && !saved.valid);
}
static int parse(int argc, char ** argv, struct options * o) { optind=0; return parse_options(argc,argv,o); }
int main(void) {
    struct options o={0};
    char * one[]={"test","--slot","5"}; assert(parse(3,one,&o)==0 && o.slot_count==1 && o.slots[0]==5);
    char * many[]={"test","--slots","1,2,5","--seconds","0.05"}; assert(parse(5,many,&o)==0);
    const char * bad[]={"",",1","1,","1,,2","1,1","6","-1"," 1","1 2","1x","01","1,2,3,4,5,0"};
    for (unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);++i) assert(parse_slots(bad[i],&o)!=0);
    char * repeated[]={"test","--slot","1","--slots","2"}; assert(parse(5,repeated,&o)!=0);
    char * twice[]={"test","--slots","1","--slots","2"}; assert(parse(5,twice,&o)!=0);
    assert(parse(5,many,&o)==0);
    reset(); assert(check_idle(ram)==0); ram[REG_SAMPLE_INTERVAL_CTL/4]=0; assert(check_idle(ram)==0);
    for (unsigned bit=0;bit<2;++bit) { ram[REG_SAMPLE_INTERVAL_CTL/4]=1U<<bit; assert(check_idle(ram)!=0); }
    reset();ram[REG_CONTROL/4]|=CTRL_METRIC_ENABLE;assert(check_idle(ram)!=0);
    reset();ram[REG_INTERRUPT_ENABLE/4]=1;assert(check_idle(ram)!=0);
    reset();ram[REG_GLOBAL_INTERRUPT/4]=1;assert(check_idle(ram)!=0);
    reset();ram[REG_INTERRUPT_STATUS/4]=1;assert(check_idle(ram)!=0);
    for (unsigned bit=0;bit<32;++bit) {
        if (!(CTRL_BUSY_MASK & (UINT32_C(1)<<bit))) continue;
        reset();ram[REG_CONTROL/4]|=UINT32_C(1)<<bit;
        assert(lifecycle(o)!=0 && writes==0 && !saved.valid);
    }
    { const uint32_t busy_regs[]={REG_GLOBAL_INTERRUPT,REG_INTERRUPT_ENABLE,REG_INTERRUPT_STATUS};
      for (unsigned i=0;i<3;++i) {
          reset();ram[busy_regs[i]/4]=1;
          assert(lifecycle(o)!=0 && writes==0 && !saved.valid);
      }
    }
    reset();save_state(ram); assert(configure_selectors(ram,&o)==0);
    assert(ram[REG_METRIC_SELECTOR0/4]==0x43422322 && ram[REG_METRIC_SELECTOR1/4]==0x1122a3a2);
    assert(ram[REG_METRIC_SELECTOR2/4]==0x55667788 && selector_writes==2); restore_state(ram);restored();
    reset(); assert(lifecycle(o)==0); restored(); assert(starts==1 && stops==1 && reads==6 && unmapped==1);
    for (unsigned i=0;i<6;++i) assert(read_offsets[i]==0x100+16*i);
    assert(strstr(output,"SELECTED_AGGREGATE scope=selected_ports_system_traffic ports=3"));
    assert(strstr(output,"write_bytes=12884901885 read_bytes=12884901885 combined_bytes=25769803770"));
    { const char * line=strstr(output,"SELECTED_AGGREGATE"); double active,w,r,c;
      assert(sscanf(strstr(line,"active_s="),"active_s=%lf",&active)==1);
      assert(sscanf(strstr(line,"write_MBps="),"write_MBps=%lf read_MBps=%lf combined_MBps=%lf",&w,&r,&c)==3);
      assert(fabs(w-12884901885.0/active/1e6)<0.001 && fabs(r-w)<0.001 && fabs(c-2*w)<0.002); }
    for (unsigned i=0;i<10;++i) { reset();inject_status=1U<<(i+3);assert(lifecycle(o)!=0);restored();assert(!strstr(output,"AGGREGATE")); }
    reset();inject_status=1;assert(lifecycle(o)!=0);restored();assert(!strstr(output,"AGGREGATE"));
    for (int i=1;i<=4;++i) { reset();bad_time=i;assert(lifecycle(o)!=0);restored();assert(!strstr(output,"AGGREGATE")); }
    for (int i=1;i<=8;++i) { reset();clock_fail=i;assert(lifecycle(o)!=0);restored();assert(!strstr(output,"AGGREGATE")); }
    reset();interrupt_sleep=1;assert(lifecycle(o)!=0);restored();assert(starts==1 && stops==1 && reads==0);
    reset();sleep_fail=1;assert(lifecycle(o)!=0);restored();assert(starts==1 && stops==1);
    reset();signal_setup=1;assert(lifecycle(o)!=0);restored();assert(starts==0);
    reset();selector_fail=1;assert(lifecycle(o)!=0);restored();assert(starts==0);
    for (int i=2;i<=8;++i) { reset();print_fail_at=i;assert(lifecycle(o)!=0);restored(); }
    for (int i=1;i<=3;++i) { reset();flush_fail_at=i;assert(lifecycle(o)!=0);restored(); }
    reset();o.seconds=NAN;assert(lifecycle(o)!=0);restored();assert(!strstr(output,"AGGREGATE"));o.seconds=0.05;
    reset();print_fail=1;assert(lifecycle(o)!=0);restored();assert(writes==0);
    reset();params[5]=5;assert(lifecycle(o)!=0);restored();assert(writes==0);
    reset();map_fail=1;assert(lifecycle(o)!=0);restored();assert(writes==0);
    reset();o.seconds=0.2;assert(lifecycle(o)==0);restored();assert(starts==2 && stops==2 && reads==12);
    assert(strstr(output,"write_bytes=25769803770 read_bytes=25769803770 combined_bytes=51539607540"));
    { const char * cursor=output; double summed=0.0, active,w,r,c; unsigned found=0;
      while ((cursor=strstr(cursor," slot=1 window="))!=NULL) {
          assert(sscanf(strstr(cursor,"active_s="),"active_s=%lf",&active)==1); summed+=active; ++found; ++cursor;
      }
      assert(found==2 && strstr(output,"write_bytes=8589934590 read_bytes=8589934590"));
      const char * line=strstr(output,"SELECTED_AGGREGATE");
      assert(sscanf(strstr(line,"active_s="),"active_s=%lf",&active)==1 && fabs(active-summed)<1e-8);
      assert(sscanf(strstr(line,"write_MBps="),"write_MBps=%lf read_MBps=%lf combined_MBps=%lf",&w,&r,&c)==3);
      assert(fabs(w-25769803770.0/summed/1e6)<0.001 && fabs(r-w)<0.001 && fabs(c-2*w)<0.002);
      for (unsigned port=0;port<o.slot_count;++port) {
          char tag[64]; snprintf(tag,sizeof(tag),"AGGREGATE slot=%u ",o.slots[port]);
          const char * per_port=strstr(output,tag);
          assert(per_port);
          assert(sscanf(strstr(per_port,"active_s="),"active_s=%lf",&active)==1 && fabs(active-summed)<1e-8);
          assert(sscanf(strstr(per_port,"write_MBps="),"write_MBps=%lf read_MBps=%lf combined_MBps=%lf",&w,&r,&c)==3);
          assert(fabs(w-8589934590.0/summed/1e6)<0.001 && fabs(r-8589934590.0/summed/1e6)<0.001);
          assert(fabs(c-17179869180.0/summed/1e6)<0.001);
      }
    }
    assert(parse_slots("5,4,3,2,1",&o)==0);reset();save_state(ram);assert(configure_selectors(ram,&o)==0);
    assert(ram[REG_METRIC_SELECTOR0/4]==0x8382a3a2 && ram[REG_METRIC_SELECTOR1/4]==0x43426362);
    assert(ram[REG_METRIC_SELECTOR2/4]==0x55662322);restore_state(ram);restored();
    assert(parse(3,one,&o)==0);o.seconds=0.05;reset();assert(lifecycle(o)==0);restored();
    assert(starts==1 && stops==1 && reads==2 && strstr(output,"AGGREGATE slot=5"));
    assert(!strstr(output,"SELECTED_AGGREGATE"));
    puts("PASS: simultaneous selectors, shared window/stop, totals/timing, ownership and cleanup (RAM mocks)");
}
'''
with tempfile.TemporaryDirectory(prefix='fpga-ddr-bandwidth-') as tmp:
    c = Path(tmp) / 'test.c'
    exe = Path(tmp) / ('test.exe' if os.name == 'nt' else 'test')
    c.write_text(code)
    subprocess.run([os.environ.get('CC', 'gcc'), '-std=c11', '-Wall', '-Wextra', '-Werror', str(c), '-o', str(exe), '-lm'], check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=30)
