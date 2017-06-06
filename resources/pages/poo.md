## Object-oriented programming!

**Object-oriented programming should be the single programming paradigm used**. Avoiding as far as possible, imperative and functional programming.

![jMetal architecture](../../resources/python_poo_programming.png)
![jMetal architecture](../../resources/python_functional_programming.png)
![jMetal architecture](../../resources/python_imperative_programming.png)

In classes, we directly access the attributes, which are usually defined as public.

![jMetal architecture](../../resources/without_getter_setter.png)

Only when we want to **implement additional logic in the accesses to the attributes** we define getter/setter methods, but **always by using the ***property*** annotation or the ***property*** function**:

![jMetal architecture](../../resources/property_annotation.png)
![jMetal architecture](../../resources/property_functional.png)

By using ***property***, we continue to access the attributes directly:

![jMetal architecture](../../resources/good_access.png)

Do not use getter/setter methods without the *property* annotation or the *property* function:

![jMetal architecture](../../resources/with_getter_setter.png)

Since this way of accessing the attribute is not commonly used in Python:

![jMetal architecture](../../resources/bad_access.png)