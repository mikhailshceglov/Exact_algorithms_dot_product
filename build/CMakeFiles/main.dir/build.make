# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/misha/leti/sem4/CM/lab2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/misha/leti/sem4/CM/lab2/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: /home/misha/leti/sem4/CM/lab2/src/main.cpp
CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/main.cpp.o -MF CMakeFiles/main.dir/src/main.cpp.o.d -o CMakeFiles/main.dir/src/main.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/kobbelt.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/kobbelt.cpp.o: /home/misha/leti/sem4/CM/lab2/src/kobbelt.cpp
CMakeFiles/main.dir/src/kobbelt.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/kobbelt.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/kobbelt.cpp.o -MF CMakeFiles/main.dir/src/kobbelt.cpp.o.d -o CMakeFiles/main.dir/src/kobbelt.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/kobbelt.cpp

CMakeFiles/main.dir/src/kobbelt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/kobbelt.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/kobbelt.cpp > CMakeFiles/main.dir/src/kobbelt.cpp.i

CMakeFiles/main.dir/src/kobbelt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/kobbelt.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/kobbelt.cpp -o CMakeFiles/main.dir/src/kobbelt.cpp.s

CMakeFiles/main.dir/src/long_accumulator.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/long_accumulator.cpp.o: /home/misha/leti/sem4/CM/lab2/src/long_accumulator.cpp
CMakeFiles/main.dir/src/long_accumulator.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/src/long_accumulator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/long_accumulator.cpp.o -MF CMakeFiles/main.dir/src/long_accumulator.cpp.o.d -o CMakeFiles/main.dir/src/long_accumulator.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/long_accumulator.cpp

CMakeFiles/main.dir/src/long_accumulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/long_accumulator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/long_accumulator.cpp > CMakeFiles/main.dir/src/long_accumulator.cpp.i

CMakeFiles/main.dir/src/long_accumulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/long_accumulator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/long_accumulator.cpp -o CMakeFiles/main.dir/src/long_accumulator.cpp.s

CMakeFiles/main.dir/src/sorting.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/sorting.cpp.o: /home/misha/leti/sem4/CM/lab2/src/sorting.cpp
CMakeFiles/main.dir/src/sorting.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/src/sorting.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/sorting.cpp.o -MF CMakeFiles/main.dir/src/sorting.cpp.o.d -o CMakeFiles/main.dir/src/sorting.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/sorting.cpp

CMakeFiles/main.dir/src/sorting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/sorting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/sorting.cpp > CMakeFiles/main.dir/src/sorting.cpp.i

CMakeFiles/main.dir/src/sorting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/sorting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/sorting.cpp -o CMakeFiles/main.dir/src/sorting.cpp.s

CMakeFiles/main.dir/src/pichat.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/pichat.cpp.o: /home/misha/leti/sem4/CM/lab2/src/pichat.cpp
CMakeFiles/main.dir/src/pichat.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/src/pichat.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/pichat.cpp.o -MF CMakeFiles/main.dir/src/pichat.cpp.o.d -o CMakeFiles/main.dir/src/pichat.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/pichat.cpp

CMakeFiles/main.dir/src/pichat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/pichat.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/pichat.cpp > CMakeFiles/main.dir/src/pichat.cpp.i

CMakeFiles/main.dir/src/pichat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/pichat.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/pichat.cpp -o CMakeFiles/main.dir/src/pichat.cpp.s

CMakeFiles/main.dir/src/fma.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/fma.cpp.o: /home/misha/leti/sem4/CM/lab2/src/fma.cpp
CMakeFiles/main.dir/src/fma.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/main.dir/src/fma.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/fma.cpp.o -MF CMakeFiles/main.dir/src/fma.cpp.o.d -o CMakeFiles/main.dir/src/fma.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/fma.cpp

CMakeFiles/main.dir/src/fma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/fma.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/fma.cpp > CMakeFiles/main.dir/src/fma.cpp.i

CMakeFiles/main.dir/src/fma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/fma.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/fma.cpp -o CMakeFiles/main.dir/src/fma.cpp.s

CMakeFiles/main.dir/src/merge.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/merge.cpp.o: /home/misha/leti/sem4/CM/lab2/src/merge.cpp
CMakeFiles/main.dir/src/merge.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/main.dir/src/merge.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/merge.cpp.o -MF CMakeFiles/main.dir/src/merge.cpp.o.d -o CMakeFiles/main.dir/src/merge.cpp.o -c /home/misha/leti/sem4/CM/lab2/src/merge.cpp

CMakeFiles/main.dir/src/merge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/merge.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/misha/leti/sem4/CM/lab2/src/merge.cpp > CMakeFiles/main.dir/src/merge.cpp.i

CMakeFiles/main.dir/src/merge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/merge.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/misha/leti/sem4/CM/lab2/src/merge.cpp -o CMakeFiles/main.dir/src/merge.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cpp.o" \
"CMakeFiles/main.dir/src/kobbelt.cpp.o" \
"CMakeFiles/main.dir/src/long_accumulator.cpp.o" \
"CMakeFiles/main.dir/src/sorting.cpp.o" \
"CMakeFiles/main.dir/src/pichat.cpp.o" \
"CMakeFiles/main.dir/src/fma.cpp.o" \
"CMakeFiles/main.dir/src/merge.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/src/kobbelt.cpp.o
main: CMakeFiles/main.dir/src/long_accumulator.cpp.o
main: CMakeFiles/main.dir/src/sorting.cpp.o
main: CMakeFiles/main.dir/src/pichat.cpp.o
main: CMakeFiles/main.dir/src/fma.cpp.o
main: CMakeFiles/main.dir/src/merge.cpp.o
main: CMakeFiles/main.dir/build.make
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/misha/leti/sem4/CM/lab2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/misha/leti/sem4/CM/lab2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/misha/leti/sem4/CM/lab2 /home/misha/leti/sem4/CM/lab2 /home/misha/leti/sem4/CM/lab2/build /home/misha/leti/sem4/CM/lab2/build /home/misha/leti/sem4/CM/lab2/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

