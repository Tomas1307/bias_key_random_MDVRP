@echo off
REM Script de compilación para Windows
echo Compilando benchmark_search_simple.cpp...

REM Intenta compilar con pthreads
g++ -std=c++11 -Wall -Wextra -O2 brkga_mdvrp.cpp benchmark_search_simple.cpp -o benchmark_search.exe -pthread

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Falló la compilación con pthreads. Intentando compilar sin multithreading...
    
    REM Versión sin multithreading (sólo para pruebas)
    g++ -std=c++11 -Wall -Wextra -O2 brkga_mdvrp.cpp benchmark_search_simple.cpp -o benchmark_search_single.exe -D_DISABLE_THREADING
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo Error: No se pudo compilar. Verifique que tiene instalado un compilador C++ compatible.
        exit /b 1
    ) else (
        echo.
        echo Compilación exitosa sin multithreading! El ejecutable es benchmark_search_single.exe
        echo.
        echo NOTA: Esta versión no utiliza múltiples hilos y será más lenta.
        echo.
        echo Para ejecutar:
        echo benchmark_search_single.exe --data-dir ./dat --instances-file ./instances_selection/benchmark_instances.txt --results-dir ./hp_results
    )
) else (
    echo.
    echo Compilación exitosa! El ejecutable es benchmark_search.exe
    echo.
    echo Para ejecutar:
    echo benchmark_search.exe --data-dir ./dat --instances-file ./instances_selection/benchmark_instances.txt --results-dir ./hp_results --threads 4
    echo.
)

exit /b 0