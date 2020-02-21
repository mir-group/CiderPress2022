

! module, function, program, or block
! 3xN, not Nx3
function hartree_potential (vh, rho_data, coords, weights, ngrid, ndat)

    implicit none

    integer                                         :: ngrid
    integer                                         :: ndat
    integer                                         :: i
    integer                                         :: j
    integer                                         :: hartree_potential
    real(8)                                         :: pi = 4 * atan(1.0_8)
    real(8), dimension(ndat,ngrid), intent(in)      :: rho_data
    real(8), dimension(3,ngrid), intent(in)         :: coords
    real(8), dimension(ngrid), intent(in)           :: weights
    real(8), dimension(ndat,ngrid), intent(out)     :: vh
    real(8), dimension(3,ngrid)                     :: vecs
    real(8), dimension(ndat,ngrid)                  :: tmp
    real(8), dimension(ngrid)                       :: rs

    hartree_potential = -1
    do i = 1, ngrid
        do j = 1, ngrid
            vecs(:,j) = coords(:,j) - coords(:,i)
        enddo
        rs(:) = norm2(vecs, 1)
        rs(i) = (2.0/3) * (3.0 * abs(weights(i)) / (4 * pi))**(1.0 / 3)
        do j=1, ndat
            tmp(j,:) = rho_data(j,:) / rs
            vh(j,i) = dot_product(tmp(j,:), weights)
            !vh(j,i) = vh(j,i) - rho_data(j,i) / rs(i) * weights(i)
        enddo
    enddo
    hartree_potential = 0

end function hartree_potential

function nonlocal_dft_data (nlc_data, rho_data, dtau_data, dvh_data,&
                            ws_radii, coords, weights, ngrid)

    implicit none

    integer                                         :: ngrid
    integer                                         :: i
    integer                                         :: j
    integer                                         :: nonlocal_dft_data
    real(8)                                         :: pi = 4 * atan(1.0_8)
    real(8), dimension(4,ngrid), intent(in)         :: rho_data
    real(8), dimension(3,ngrid), intent(in)         :: dtau_data
    real(8), dimension(3,ngrid), intent(in)         :: dvh_data
    real(8), dimension(ngrid), intent(in)           :: ws_radii
    real(8), dimension(3,ngrid), intent(in)         :: coords
    real(8), dimension(ngrid), intent(in)           :: weights
    real(8), dimension(5,ngrid), intent(out)        :: nlc_data
    real(8), dimension(3,ngrid)                     :: vecs
    real(8), dimension(ngrid)                       :: tmp
    real(8), dimension(ngrid)                       :: rs
    real(8), dimension(ngrid)                       :: exp_weights

    nonlocal_dft_data = -1
    nlc_data(1,:) = norm2(dvh_data(1:3,:))
    do i = 1, ngrid
        do j = 1, ngrid
            vecs(:,j) = coords(:,j) - coords(:,i)
        enddo
        rs(:) = norm2(vecs, 1)
        exp_weights(:) = exp(-rs / ws_radii(i)) * weights * rho_data(1,:)
        nlc_data(5,i) = sum(exp_weights)
        tmp(:) = sum(vecs * rho_data(2:4,:), 1)
        nlc_data(3,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dtau_data(1:3,:))
        nlc_data(4,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dvh_data(1:3,:))
        nlc_data(2,i) = dot_product(tmp, exp_weights)
    enddo
    nonlocal_dft_data = 0

end function nonlocal_dft_data
    