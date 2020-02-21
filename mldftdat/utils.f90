

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
        rs(i) = (2.0/3) * (3.0 * weights(i) / (4 * pi))**(1.0 / 3)
        do j=1, ndat
            tmp(j,:) = rho_data(j,:) / rs
            vh(j,i) = dot_product(tmp(j,:), weights)
        enddo
    enddo
    hartree_potential = 0

end function hartree_potential

!function nonlocal_dft_data ()

!    implicit none

!    real(8) :: 
!    integer :: ngrid

!    do i = 1, ngrid
!        vecs(:,:) = coords(:,:) - coords(i,:)
!        rs = norm2(vecs, 1)
!        exp_weights(:) = exp(-rs / ws_radii(i)) * weights
!        rddrho(:,:) = dot(vecs, drho, 1)
!        rddvh(:,:)  = dot(vecs, dvh, 1)
!        rddtau(:,:) = dot(vecs, dtau, 1)

!        rddrho_int = dot(exp_weights, rho * rddrho)
!        rddvh_int  = dot(exp_weights, rho * rddvh)
!        rddtau_int = dot(exp_weights, rho * rddtau)
!    enddo

!end function nonlocal_dft_data
    