import { StyleSheet } from 'react-native';
import { AppColors } from '../../utils/AppCollors';
import { height, width } from '../../methord/Dimentions';

const styles = StyleSheet.create({
  header: {
    width: width(100),
    height: height(8),
    justifyContent: 'center',
    backgroundColor: AppColors.primary,
    paddingTop: 20,
  },
  headerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,

  },
  menuIcon: {
    marginRight: 20,

  },
  userAvatar: {

    width: 60,
    height: 60,
    borderRadius: 30,
    marginRight: 25,
  },
  // userInfo: {
  //   justifyContent: 'center',
  //   textAlign: 'center',
  // },
  detectionText: {
    width: width(60),
    textAlign: 'center',
    fontSize: 16,
    fontWeight: '200',
    color: '#fff',
  },
  userName: {
    fontSize: 19,
    fontWeight: 'bold',
    color: '#fff',
  },
});
export default styles