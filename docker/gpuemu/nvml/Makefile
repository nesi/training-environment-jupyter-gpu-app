# Builds the stand-in libnvidia-ml.so.1.
#
# The SONAME matters: nvtop and friends dlopen("libnvidia-ml.so.1"), and the
# dynamic linker matches on SONAME rather than filename, so this has to be
# baked in rather than left to a symlink.

CC      ?= cc
PREFIX  ?= /opt/gpuemu
LIBDIR  ?= $(PREFIX)/lib

CFLAGS  ?= -O2 -g
CFLAGS  += -Wall -Wextra -Werror -std=c11 -fPIC -fvisibility=hidden
LDFLAGS += -shared -Wl,-soname,libnvidia-ml.so.1 -lpthread

TARGET   = libnvidia-ml.so.1
SOURCES  = libnvidia_ml.c
HEADERS  = gpuemu_shm.h

.PHONY: all install clean abi

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

# Prints the struct sizes the C side expects, so tests/test_abi.py can check
# that gpuemu/shm.py agrees with this header rather than assuming it does.
abi: abi_dump
	./abi_dump

abi_dump: abi_dump.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ abi_dump.c

install: $(TARGET)
	install -d $(DESTDIR)$(LIBDIR)
	install -m 0755 $(TARGET) $(DESTDIR)$(LIBDIR)/$(TARGET)
	ln -sf $(TARGET) $(DESTDIR)$(LIBDIR)/libnvidia-ml.so

clean:
	rm -f $(TARGET) abi_dump
