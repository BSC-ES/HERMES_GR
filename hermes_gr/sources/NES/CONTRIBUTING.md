# Contributing

1.  Create an issue and complete the description. Complete the issue
    description as much as possible with (estimated time, corresponding
    milestone, assigned person, etc.)
2.  Create a branch (from master) directly in the issue. Its name should
    start with `develop-`, followed by the number and title of the issue
    that appears by default.
3.  Clone and checkout to the new branch, without any modification, in
    Nord3v2. It is recommended to run some tests to ensure the current
    behavior.
4.  Modify the code.
5.  Run the simulation with the new branch: It is important to prepend
    the cloned path in the `PYTHONPATH`, e.g.
    `export PYTHONPATH=/gpfs/scratch/bsc32/bsc32538/nes:${PYTHONPATH}`.
6.  Create and run a specific test for your case in the folder `tests`.
7.  Update the `CHANGELOG.md` and include information on the new
    development or bug fix.
8.  Update the wiki with the new specifications.
9.  Merge `master` into your development branch. To ensure that if there
    has been any changes in the master branch, these are included.
10. Run all tests in `tests`.
11. Create a merge request and assign it to Alba
    ([@avilanov](https://earth.bsc.es/gitlab/avilanov)) or Carles
    ([@ctena](https://earth.bsc.es/gitlab/ctena)), who will review it.
