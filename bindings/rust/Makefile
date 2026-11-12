.DELETE_ON_ERROR:

deps: sqlite-vec.c sqlite-vec.h sqlite3ext.h sqlite3.h LICENSE-MIT LICENSE-APACHE

sqlite-vec.c: ../../sqlite-vec.c $(wildcard ../../sqlite-vec-*.c) ../../scripts/amalgamate.py
		python3 ../../scripts/amalgamate.py $< > $@

sqlite-vec.h: ../../sqlite-vec.h
		cp $< $@

sqlite3ext.h: ../../vendor/sqlite3ext.h
		cp $< $@

sqlite3.h: ../../vendor/sqlite3.h
		cp $< $@

LICENSE-MIT: ../../LICENSE-MIT
		cp $< $@

LICENSE-APACHE: ../../LICENSE-APACHE
		cp $< $@

.PHONY: deps
