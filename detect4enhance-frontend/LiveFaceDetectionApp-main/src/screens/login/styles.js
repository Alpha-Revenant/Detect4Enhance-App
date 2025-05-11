import { StyleSheet } from 'react-native';
import { height, width } from '../../methord/Dimentions';
import { AppColors } from '../../utils/AppCollors';

const styles = StyleSheet.create({
  container: {
    width: width(100),
    height: height(100),
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputContainer: {
    alignSelf: 'center',
    width: width(90),
    marginBottom: height(1),
    marginTop: 20
  },
  eyeIcon: {
    width: width(6),
    // height: height(5),
    // backgroundColor: 'red',
    justifyContent: 'center',
    marginRight: width(10),

  },
  label: {
    marginLeft: width(1),
    marginBottom: 10,
    fontWeight: 'bold',
  },
  passwordInputContainer: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: '#3C3C3C',
    borderRadius: 5,
  },
  input: {
    // backgroundColor: 'yellow',
    width: width(80),

    padding: 10,
    // justifyContent: 'space-between',
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginTop: 2,
    marginRight: width(5)
  },
  forgotPasswordText: {
    color: AppColors.primary,
  },
  loginButton: {
    backgroundColor: AppColors.primary,
    // ,
    width: width(90),
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    marginTop: 30,
    elevation: 10
  },
  loginButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  signupButton: {
    backgroundColor: '#fff',
    width: width(90),
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    marginTop: 30,
    elevation: 10
  },
  signupButtonText: {
    color: 'black',
    fontWeight: 'bold',
    fontSize: 16,
  },
  error: {
    alignSelf: 'flex-end',
    flexDirection: "row",
    marginTop: 7
  },
});

export default styles;
