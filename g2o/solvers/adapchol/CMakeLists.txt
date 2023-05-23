add_subdirectory(adapchol/host)



add_library(solver_adapchol ${G2O_LIB_TYPE}
        solver_adapchol.cpp
        linear_solver_adapchol.h
        )

set_target_properties(solver_adapchol PROPERTIES OUTPUT_NAME ${LIB_PREFIX}solver_adapchol)
set_target_properties(solver_adapchol PROPERTIES
        VERSION ${G2O_LIB_VERSION}
        SOVERSION ${G2O_LIB_SOVERSION})
if (APPLE)
    set_target_properties(solver_adapchol PROPERTIES INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()
target_compile_features(solver_adapchol PUBLIC cxx_std_17)
target_link_libraries(solver_adapchol core adapchol_host)

target_include_directories(solver_adapchol PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/g2o/solvers/adapchol>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/adapchol/host/include>
        )

install(TARGETS solver_adapchol adapchol_host adapchol_cpu_backend adapchol_fpga_backend csparse internal_io cs_adap
        EXPORT ${G2O_TARGETS_EXPORT_NAME}
        RUNTIME DESTINATION ${RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
        INCLUDES DESTINATION ${INCLUDES_DESTINATION}
        )

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
install(FILES ${headers} DESTINATION ${INCLUDES_INSTALL_DIR}/solvers/adapchol)

