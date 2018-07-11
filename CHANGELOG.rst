
Changelog
=========

1.1.0 (in progress)
------------------

- Implement discrete time systems using events.
    - Since systems with discontinuities should have been continuous-time
      anyway, the ``dt`` argument to the numeric ``SwitchedSystem`` constructor
      was removed. 
    - The argument rules for ``event_equation_function`` and 
      ``update_equation_function`` during integration now follow 
      ``output_equation_function`` and ``state_equation_function``,
      respectively.
    - No longer allow non-zero t0 for simulate; 


1.0.0 (2017-08-29)
------------------

- Public release.
