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
include examples/CMakeFiles/example_learned.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/example_learned.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/example_learned.dir/flags.make

examples/CMakeFiles/example_learned.dir/example_learned.cpp.o: examples/CMakeFiles/example_learned.dir/flags.make
examples/CMakeFiles/example_learned.dir/example_learned.cpp.o: ../examples/example_learned.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/wlouyang/Downloads/gop_1.3/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/example_learned.dir/example_learned.cpp.o"
	cd /home/wlouyang/Downloads/gop_1.3/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/example_learned.dir/example_learned.cpp.o -c /home/wlouyang/Downloads/gop_1.3/examples/example_learned.cpp

examples/CMakeFiles/example_learned.dir/example_learned.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_learned.dir/example_learned.cpp.i"
	cd /home/wlouyang/Downloads/gop_1.3/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/wlouyang/Downloads/gop_1.3/examples/example_learned.cpp > CMakeFiles/example_learned.dir/example_learned.cpp.i

examples/CMakeFiles/example_learned.dir/example_learned.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_learned.dir/example_learned.cpp.s"
	cd /home/wlouyang/Downloads/gop_1.3/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/wlouyang/Downloads/gop_1.3/examples/example_learned.cpp -o CMakeFiles/example_learned.dir/example_learned.cpp.s

examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.requires:
.PHONY : examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.requires

examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.provides: examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/example_learned.dir/build.make examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.provides.build
.PHONY : examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.provides

examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.provides.build: examples/CMakeFiles/example_learned.dir/example_learned.cpp.o

# Object files for target example_learned
example_learned_OBJECTS = \
"CMakeFiles/example_learned.dir/example_learned.cpp.o"

# External object files for target example_learned
example_learned_EXTERNAL_OBJECTS =

examples/example_learned: examples/CMakeFiles/example_learned.dir/example_learned.cpp.o
examples/example_learned: examples/CMakeFiles/example_learned.dir/build.make
examples/example_learned: lib/imgproc/libimgproc.a
examples/example_learned: lib/proposals/libproposals.a
examples/example_learned: lib/contour/libcontour.a
examples/example_learned: lib/segmentation/libsegmentation.a
examples/example_learned: lib/contour/libcontour.a
examples/example_learned: lib/imgproc/libimgproc.a
examples/example_learned: /usr/lib64/libjpeg.so
examples/example_learned: /usr/lib64/libpng.so
examples/example_learned: /usr/lib64/libz.so
examples/example_learned: lib/learning/liblearning.a
examples/example_learned: lib/util/libutil.a
examples/example_learned: external/liblbfgs-1.10/liblbfgs.a
examples/example_learned: examples/CMakeFiles/example_learned.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable example_learned"
	cd /home/wlouyang/Downloads/gop_1.3/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_learned.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/example_learned.dir/build: examples/example_learned
.PHONY : examples/CMakeFiles/example_learned.dir/build

examples/CMakeFiles/example_learned.dir/requires: examples/CMakeFiles/example_learned.dir/example_learned.cpp.o.requires
.PHONY : examples/CMakeFiles/example_learned.dir/requires

examples/CMakeFiles/example_learned.dir/clean:
	cd /home/wlouyang/Downloads/gop_1.3/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/example_learned.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/example_learned.dir/clean

examples/CMakeFiles/example_learned.dir/depend:
	cd /home/wlouyang/Downloads/gop_1.3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wlouyang/Downloads/gop_1.3 /home/wlouyang/Downloads/gop_1.3/examples /home/wlouyang/Downloads/gop_1.3/build /home/wlouyang/Downloads/gop_1.3/build/examples /home/wlouyang/Downloads/gop_1.3/build/examples/CMakeFiles/example_learned.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/example_learned.dir/depend

