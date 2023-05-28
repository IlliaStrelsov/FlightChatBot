class FlightOrderDataManager:
    date = ""
    cityFrom = ""
    cityTo = ""
    costs = ""
    name = ""

    def getAvailableOrders(self):
        if self.cityFrom and self.cityTo and self.date:
            # Here should be logic of getting flight info from dest and date
            return "Some flight info"
