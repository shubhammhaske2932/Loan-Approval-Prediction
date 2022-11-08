def get_gender_encode(Gender):
    if Gender == 'Male':
        Gender = 1
    elif Gender == 'Female':
        Gender = 0
    else:
        Gender = "Please enter correct value"
        
    return Gender


def get_married_encode(Married):
        
    if Married == 'Yes':
        Married = 1
    elif Married == 'No':
        Married = 0
    else:
        Married = "Please enter correct value"
        
    return Married


def get_dependents_encode(Dependents):
    if Dependents == '0':
        Dependents = 0
    elif Dependents == '1':
        Dependents = 1
    elif Dependents == '2':
        Dependents = 2
    elif Dependents == '3+' or Dependents == '3':
        Dependents = 3
    else:
        Dependents = "Please enter correct value"
        
    return Dependents


def get_education_encode(Education):
    if Education == 'Graduate':
        Education = 1
    elif Education == 'Not Graduate':
        Education = 0
    else:
        Education = "Please enter correct value"
        
    return Education


def get_self_emp_encode(Self_Employed):
    if Self_Employed == 'Yes':
        Self_Employed = 1
    elif Self_Employed == 'No':
        Self_Employed = 0
    else:
        Self_Employed = "Please enter correct value"
        
    return Self_Employed


def get_property_area_encode(Property_Area):
    if Property_Area == 'Urban':
        Property_Area = 0
    elif Property_Area == 'Semiurban':
        Property_Area = 1
    elif Property_Area == 'Rural':
        Property_Area = 2
    else:
        Property_Area = "Please enter correct value"
        
    return Property_Area