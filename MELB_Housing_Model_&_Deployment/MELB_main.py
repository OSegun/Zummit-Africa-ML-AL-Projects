from pydantic import BaseModel

class Housedata(BaseModel):
    Suburb:          object 
    Address:         object
    Rooms:           int  
    Type:            int  
    Method:          int 
    SellerG:         object
    Date:            object 
    Distance:        float
    Postcode:        float
    Bedroom2:        float
    Bathroom:        float
    Car:             float
    Landsize:        float
    BuildingArea:    float
    YearBuilt:       float
    CouncilArea:     object
    Lattitude:       float
    Longtitude:      float
    Regionname:      int  
    Propertycount:   float
