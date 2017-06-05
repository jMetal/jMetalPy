## Python 3.5!

We use the new features of python 3. Concretely, up to version **3.5**.

### Typing

We **always** define types in the parameters of the arguments and the return value:

![jMetal architecture](resources/types_in_methods.png)

### Abstract class

We can define abstract classes (ABCs) in Python:

![jMetal architecture](resources/abstract.png)

In the case that we want to define an **interface** class, it is done in the same way. We just have to define all the methods of the class as abstract.

### Generic classes and types

Example of use of generic types:

![jMetal architecture](resources/generic_types.png)

In the code below, the IDE displays a **warning**, since although the 2nd parameter is a float type, which is a type allowed in the definition of the generic type X, it is not of the same type as the first, since the first 2 parameters must be of the same generic type (S):

![jMetal architecture](resources/instance_with_generic_types1_wearning.png)

In the code below, the IDE displays a **warning**, since the 2nd parameter is a type not allowed in the definition of the generic type ( *TypeVar('S', int, float)* ):

![jMetal architecture](resources/instance_with_generic_types2_wearning.png)

Example of use of **generic class**. When the class inherits from *Generic[...]*, the class is defined as generic. In this way we can indicate the types that will have the values of the generic types, when using the class as type. Look at the *add_car()* method of the *Parking* class.

NOTE: The generic classes inherit from abc.ABCMeta, so they are abstract classes and **abstract methods can be used** .

![jMetal architecture](resources/generic_class1.png)
![jMetal architecture](resources/generic_class2.png)

In the code below, the IDE displays a warning in the call to the *add_car()* method when adding the car, since the 3rd parameter of the init must be a *str* type, as defined in the *add_car()* method of the *Parking* class.

![jMetal architecture](resources/instance_with_generic_class_wearning.png)

When inheriting from generic classes, some type variables could be fixed:

![jMetal architecture](resources/generic_types_fixed.png)

Inheritance from non-generic class to generic class:

![jMetal architecture](resources/inheritance_non_generic_to_generic.png)

Inheritance from generic class to another generic class:

![jMetal architecture](resources/inheritance_generic_to_generic.png)
