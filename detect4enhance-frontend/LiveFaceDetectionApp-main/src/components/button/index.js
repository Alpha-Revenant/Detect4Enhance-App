import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { AppColors } from '../../utils/AppCollors';
import { width } from '../../methord/Dimentions';

export default function Button({ label, onPress, style, ...rest }) {
    return (
        <TouchableOpacity style={[styles.loginButton, style]} onPress={onPress} {...rest}>
            <Text style={styles.loginButtonText}>{label}</Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
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
})
