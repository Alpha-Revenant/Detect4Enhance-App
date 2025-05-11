import { StyleSheet } from 'react-native';
import { AppColors } from '../../utils/AppCollors';
import { height, width } from '../../methord/Dimentions';

const styles = StyleSheet.create({
  drawerContainer: {
    flex: 1,
    backgroundColor: AppColors.primary,
    // justifyContent: "space-between",
  },
  logo: {
    width: width(30),
    height: height(15),
    borderRadius: 10,
    alignSelf: "center",
  },
  header: {
    marginTop: 20,
    padding: 28
  },

  nameText: {
    width: width(40),
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    alignSelf: "center",

  },
  content: {
    flex: 1,
    justifyContent: "flex-end",
    marginBottom: 20,
    paddingHorizontal: 16,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logoutText: {
    marginLeft: 15,
    fontSize: 16,
    color: '#fff',
  },
});
export default styles