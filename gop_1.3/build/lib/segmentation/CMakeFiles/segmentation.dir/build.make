# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wlouyang/Downloads/gop_1.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wlouyang/Downloads/gop_1.3/build

# Include any dependencies generated for this target.
include lib/segmentation/CMakeFiles/segmentation.dir/depend.make

# Include the progress variables for this target.
include lib/segmentation/CMakeFiles/segmentation.dir/progress.make

# Include the compile flags for this target's objects.
include lib/segmentation/CMakeFiles/segmentation.dir/flags.make

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o: lib/segmentation/CMakeFiles/segmentation.dir/flags.make
lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o: ../lib/segmentation/aggregation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/wlouyang/Downloads/gop_1.3/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/segmentation.dir/aggregation.cpp.o -c /home/wlouyang/Downloads/gop_1.3/lib/segmentation/aggregation.cpp

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentation.dir/aggregation.cpp.i"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/wlouyang/Downloads/gop_1.3/lib/segmentation/aggregation.cpp > CMakeFiles/segmentation.dir/aggregation.cpp.i

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentation.dir/aggregation.cpp.s"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/wlouyang/Downloads/gop_1.3/lib/segmentation/aggregation.cpp -o CMakeFiles/segmentation.dir/aggregation.cpp.s

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.requires:
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.requires

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.provides: lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.requires
	$(MAKE) -f lib/segmentation/CMakeFiles/segmentation.dir/build.make lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.provides.build
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.provides

lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.provides.build: lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o: lib/segmentation/CMakeFiles/segmentation.dir/flags.make
lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o: ../lib/segmentation/iouset.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/wlouyang/Downloads/gop_1.3/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/segmentation.dir/iouset.cpp.o -c /home/wlouyang/Downloads/gop_1.3/lib/segmentation/iouset.cpp

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentation.dir/iouset.cpp.i"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/wlouyang/Downloads/gop_1.3/lib/segmentation/iouset.cpp > CMakeFiles/segmentation.dir/iouset.cpp.i

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentation.dir/iouset.cpp.s"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/wlouyang/Downloads/gop_1.3/lib/segmentation/iouset.cpp -o CMakeFiles/segmentation.dir/iouset.cpp.s

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.requires:
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.requires

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.provides: lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.requires
	$(MAKE) -f lib/segmentation/CMakeFiles/segmentation.dir/build.make lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.provides.build
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.provides

lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.provides.build: lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o: lib/segmentation/CMakeFiles/segmentation.dir/flags.make
lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o: ../lib/segmentation/segmentation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/wlouyang/Downloads/gop_1.3/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/segmentation.dir/segmentation.cpp.o -c /home/wlouyang/Downloads/gop_1.3/lib/segmentation/segmentation.cpp

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentation.dir/segmentation.cpp.i"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/wlouyang/Downloads/gop_1.3/lib/segmentation/segmentation.cpp > CMakeFiles/segmentation.dir/segmentation.cpp.i

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentation.dir/segmentation.cpp.s"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/wlouyang/Downloads/gop_1.3/lib/segmentation/segmentation.cpp -o CMakeFiles/segmentation.dir/segmentation.cpp.s

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.requires:
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.requires

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.provides: lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.requires
	$(MAKE) -f lib/segmentation/CMakeFiles/segmentation.dir/build.make lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.provides.build
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.provides

lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.provides.build: lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o

# Object files for target segmentation
segmentation_OBJECTS = \
"CMakeFiles/segmentation.dir/aggregation.cpp.o" \
"CMakeFiles/segmentation.dir/iouset.cpp.o" \
"CMakeFiles/segmentation.dir/segmentation.cpp.o"

# External object files for target segmentation
segmentation_EXTERNAL_OBJECTS =

lib/segmentation/libsegmentation.a: lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o
lib/segmentation/libsegmentation.a: lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o
lib/segmentation/libsegmentation.a: lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o
lib/segmentation/libsegmentation.a: lib/segmentation/CMakeFiles/segmentation.dir/build.make
lib/segmentation/libsegmentation.a: lib/segmentation/CMakeFiles/segmentation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libsegmentation.a"
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && $(CMAKE_COMMAND) -P CMakeFiles/segmentation.dir/cmake_clean_target.cmake
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/segmentation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/segmentation/CMakeFiles/segmentation.dir/build: lib/segmentation/libsegmentation.a
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/build

lib/segmentation/CMakeFiles/segmentation.dir/requires: lib/segmentation/CMakeFiles/segmentation.dir/aggregation.cpp.o.requires
lib/segmentation/CMakeFiles/segmentation.dir/requires: lib/segmentation/CMakeFiles/segmentation.dir/iouset.cpp.o.requires
lib/segmentation/CMakeFiles/segmentation.dir/requires: lib/segmentation/CMakeFiles/segmentation.dir/segmentation.cpp.o.requires
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/requires

lib/segmentation/CMakeFiles/segmentation.dir/clean:
	cd /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation && $(CMAKE_COMMAND) -P CMakeFiles/segmentation.dir/cmake_clean.cmake
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/clean

lib/segmentation/CMakeFiles/segmentation.dir/depend:
	cd /home/wlouyang/Downloads/gop_1.3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wlouyang/Downloads/gop_1.3 /home/wlouyang/Downloads/gop_1.3/lib/segmentation /home/wlouyang/Downloads/gop_1.3/build /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation /home/wlouyang/Downloads/gop_1.3/build/lib/segmentation/CMakeFiles/segmentation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/segmentation/CMakeFiles/segmentation.dir/depend

