## Git WorkFlow

We have a set of branches on the remote Git server.
Some branches are temporary, and others are constant throughout the life of the repository.

* Branches always present in the repository:
> * *master*: You have the latest released to production, receive merges from the develop branch, or merge from a *hotfix* branch (emergency).
> > * Do I have to put a TAG when doing a merge from develop to master? yes
> > * Do I have to put a TAG when doing a merge from a hotfix branch to master? yes
> > * After merge from a hotfix to master, do I have to merge from master to develop? yes
> * *develop*: It is considered the "Next Release", receives merges from branches of each developer, either corrections (*fix*) or new features (*feature*).

* Temporary branches:
> * *feature/<task-id>-<description>*: When we are doing a development, we create a local branch with the prefix "feature/", then only if there is a task id, we indicate it and we add a hyphen. The following we indicate a description according to the functionality that we are developing. The words are separated by hyphens.
> > * Where does this branch emerge? This branch always emerge from the develop branch
> > * When I finish the development in my feature branch, which branch to merge into?: You always merge feature branch into develop branch

> * *fix/<task-id>-<description>*: When we are making a correction, we create a local branch with the prefix "fix/", then only if there is a task id, we indicate it and we add a hyphen. The following we indicate a description according to the functionality that we are correcting. The words are separated by hyphens.
> > * Where does this branch emerge? This branch always emerge from the develop branch
> > * When I finish the correction in my fix branch, which branch to merge into?: You always merge feature branch into develop branch

> * *hotfix/<task-id>-<description>*: Cuando estemos corrigiendo de emergencia una incidencia en producción, creamos una rama local con el prefijo "hotfix/", a continuación, le indicamos el nº de tarea que RedMine nos ha suministrado con la tarea que estemos haciendo, agregamos un guión "-" y le indicamos un nombre acorde a la funcionalidad que estemos corrigiendo.
> > * ¿De dónde sale esta rama?: Siempre sale de la rama master
> > * ¿A qué rama hago el merge cuando acabo el desarrollo?: Siempre merge a la rama master y develop

![jMetal architecture](../../resources/WorkflowGitBranches.png)

* Pasos a seguir cuando estás creando o vas a trabajar en una rama de cualquier tipo (feature / fix / hotfix):
> * Si estás creando la rama, oriéntala siempre a la funcionalidad y no metas contenido de otra tareas :)
> * 1º) Si eres quien ha creado la rama, inmediatamente después de crearla en local, súbela al servidor Git remoto, desde ese momento tienes garantizado que el sistema de integración estará verificando tu código desde el primer momento.
> * 2º) Cada vez que realices un "commit", en la medida de lo posible, envía un push al servidor, cada push desencadenará el lanzamiento automatizado de las pruebas, test, etc.
> * 3º) Una vez finalizado el desarrollo, habiendo realizado un push al servidor Git remoto, y viendo que Jenkins ha compilado y pasado la fase de test sin problema, solicita pull request a los miembros del equipo.
> * 4º) Cuando >= 75% de las personas a las que les solicitaste la pull request hayan validado la pull...
> > * Merge a la rama develop local, incluir la opción como argumento "--no--ff". Ej.: git merge feature/47-aaa --no-ff
> > * Push al servidor remoto de tu rama develop
> > * Borra la rama remota en la que estabas trabajando
> > * Borra la rama local en la que estabas trabajando

* Comandos básicos de GIT:
> * git fetch --prune: Realiza limpieza de los punteros de ramas de tu servidor local, que ya no existen en el servidor remoto
> * git -c user.name="userName" -c user.email="name@domain.es" commit -m "Comment": Permite realizar un commit incluyendo en el commit en concreto, el usuario y correo electrónico