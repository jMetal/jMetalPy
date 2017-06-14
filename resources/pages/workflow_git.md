## Git WorkFlow

We have a set of branches on the remote Git server.
Some branches are temporary, and others are constant throughout the life of the repository.

* Branches always present in the repository:
    * *master*: You have the latest released to production, receive merges from the develop branch, or merge from a *hotfix* branch (emergency).
        * Do I have to put a TAG when doing a merge from develop to master? yes
        * Do I have to put a TAG when doing a merge from a hotfix branch to master? yes
        * After merge from a hotfix to master, do I have to merge from master to develop? yes
    * *develop*: It is considered the "Next Release", receives merges from branches of each developer, either corrections (*fix*) or new features (*feature*).

* Temporary branches:
    * *feature/\<task\-id\>\-\<description\>*: When we are doing a development, we create a local branch with the prefix "feature/", then only if there is a task id, we indicate it and we add a hyphen. The following we indicate a description according to the functionality that we are developing. The words are separated by hyphens.
        * Where does this branch emerge? This branch always emerge from the develop branch
        * When I finish the development in my feature branch, which branch to merge into?: You always merge feature branch into develop branch

    * *fix/\<task\-id\>\-\<description\>*: When we are making a correction, we create a local branch with the prefix "fix/", then only if there is a task id, we indicate it and we add a hyphen. The following we indicate a description according to the functionality that we are correcting. The words are separated by hyphens.
        * Where does this branch emerge? This branch always emerge from the develop branch
        * When I finish the correction in my fix branch, which branch to merge into?: You always merge feature branch into develop branch

    * *hotfix/\<task\-id\>\-\<description\>*: When we are correcting an emergency incidence in production, we create a local branch with the prefix "hotfix/", then only if there is a task id, we indicate it and we add a hyphen. The following we indicate a description according to the functionality that we are correcting. The words are separated by hyphens.
        * Where does this branch emerge?: This branch always emerge from the master branch
        * When I finish the correction in my hotfix branch, which branch to merge into?: This branch always emerge from the master and develop branch

![jMetal architecture](../../resources/WorkflowGitBranches.png)

* Steps to follow when you are creating or going to work on a branch of any kind (feature/fix/hotfix):
    1. After you create your branch (feature/fix/hotfix) locally, upload it to the remote Git server. The integration system will verify your code from the outset.
    2. Each time you commit, as much as possible, you send a push to the server. Each push will trigger the automated launch of the tests, etc.
    3. Once the development is finished, having done a push to the remote Git server, and that the test phase has passed without problem, you create an pull request:
https://help.github.com/articles/creating-a-pull-request/
<br>NOTE:Do not forget to remove your branch (feature/fix/hotfix) once the merge has been made.

* Some useful Git commands:
    * git fetch --prune: Cleaning branches removed and bringing new branches
