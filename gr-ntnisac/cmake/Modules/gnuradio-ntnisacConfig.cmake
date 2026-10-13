find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_NTNISAC gnuradio-ntnisac)

FIND_PATH(
    GR_NTNISAC_INCLUDE_DIRS
    NAMES gnuradio/ntnisac/api.h
    HINTS $ENV{NTNISAC_DIR}/include
        ${PC_NTNISAC_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_NTNISAC_LIBRARIES
    NAMES gnuradio-ntnisac
    HINTS $ENV{NTNISAC_DIR}/lib
        ${PC_NTNISAC_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-ntnisacTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_NTNISAC DEFAULT_MSG GR_NTNISAC_LIBRARIES GR_NTNISAC_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_NTNISAC_LIBRARIES GR_NTNISAC_INCLUDE_DIRS)
